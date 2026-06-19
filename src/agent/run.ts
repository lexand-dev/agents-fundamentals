import "dotenv/config"
import { Laminar } from "@lmnr-ai/lmnr";
import { deepseek } from "@ai-sdk/deepseek"
import { generateText, streamText, type ModelMessage } from "ai"

import { tools } from "./tools/index.ts"
import { getTracer } from "@lmnr-ai/lmnr";
import { executeTool } from "./executeTools.ts"
import { SYSTEM_PROMPT } from "./system/prompt.ts"
import type { AgentCallbacks, ToolCallInfo } from "../types.ts";
import { filterCompatibleMessages } from "./system/filterMessages.ts";

Laminar.initialize({
  projectApiKey: process.env.LMNR_PROJECT_API_KEY,
})

const MODEL_NAME = "deepseek-chat";

export const runAgent = async (
  userMessage: string,
  conversationHistory: ModelMessage[],
  callbacks: AgentCallbacks
): Promise<any> => {
  const workingHistory = filterCompatibleMessages(conversationHistory)

  const messages: ModelMessage[] = [
    { role: "system", content: SYSTEM_PROMPT },
    ...workingHistory,
    { role: "user", content: userMessage },
  ];

  // await Laminar.flush();

  let fullResponse = "";

  while (true) {
    // LLM
    const result = streamText({
      model: deepseek(MODEL_NAME),
      messages,
      tools,
      experimental_telemetry: {
        isEnabled: true,
        tracer: getTracer(),
      }
    })

    const toolCalls: ToolCallInfo[] = []
    let currenText = "";
    let streamError: Error | null = null;

    try {
      for await (const chunk of result.fullStream) {
        if (chunk.type === "text-delta") {
          currenText += chunk.text;
          callbacks.onToken(chunk.text) // UI purposes to show each token
        }

        if (chunk.type === "tool-call") {
          const input = "input" in chunk ? chunk.input : {};
          toolCalls.push({
            toolCallId: chunk.toolCallId,
            toolName: chunk.toolName,
            args: input as any
          })
          callbacks.onToolCallStart(chunk.toolName, input)
        }
      }
    } catch (e: unknown) {
      streamError = e as Error;

      if (!currenText && !streamError.message.includes("No output generated")) { // AI SDK docs error-message
        throw streamError;
      }
    }

    fullResponse += currenText;

    // If we have a LLM Error
    if (streamError && !currenText) {
      fullResponse = "Sorry about that. We working on it.";
      callbacks.onToken(fullResponse);
      break;
    }

    // If not there tool calls, send a response to finish the loop
    const finishReason = await result.finishReason;

    if (finishReason !== "tool-calls" || toolCalls.length === 0) {
      const responseMessages = await result.response;
      messages.push(...responseMessages.messages);
      break;
    }

    //
    const responseMessages = await result.response;
    messages.push(...responseMessages.messages);

    // Execute tool and send back the result to the LLM in messages array
    for (const tc of toolCalls) {
      const result = await executeTool(tc.toolName, tc.args);
      callbacks.onToolCallEnd(tc.toolName, result);

      messages.push({
        role: "tool",
        content: [
          {
            type: "tool-result",
            toolCallId: tc.toolCallId,
            toolName: tc.toolName,
            output: { type: "text", value: result },
          },
        ],
      });
    }
  }

  callbacks.onComplete(fullResponse);

  return messages;
}
