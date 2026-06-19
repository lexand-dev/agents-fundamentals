import { evaluate } from "@lmnr-ai/lmnr";
import { singleTurnWithMocks } from "./executors";
import type { EvalData } from "./types";

import dataset from "./data/file-tools.json" with { type: "json" };
import { toolSelectionScore } from "./evaluators";


const executor = async (data: EvalData) => {
  return singleTurnWithMocks(data)
}

evaluate({
  data: dataset as any,
  executor,
  evaluators: {
    selectionScore: (output: any, target: any) => {
      if (target?.category === "secondary") return 1;

      return toolSelectionScore(output, target)
    }
  },
  groupName: "file-tools-selection",

})
