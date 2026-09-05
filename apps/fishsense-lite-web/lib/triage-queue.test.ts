import { describe, expect, it } from "vitest";
import { reasonKey } from "./triage-queue";

describe("reasonKey", () => {
  // Reasons carry the task id, so grouping on the raw string would produce one
  // bucket per task and the counts would all read 1 — which is what the flat
  // five-reason sample effectively did.
  it("strips the task id so identical reasons group", () => {
    expect(reasonKey("task 41822: is_labeled")).toBe("is_labeled");
    expect(reasonKey("task 9: no prediction (0 present)")).toBe(
      "no prediction (0 present)",
    );
  });

  it("groups two tasks refused for the same reason", () => {
    expect(reasonKey("task 1: is_labeled")).toBe(reasonKey("task 2: is_labeled"));
  });

  it("leaves a reason carrying no task prefix alone", () => {
    expect(reasonKey("unpublished in Label Studio")).toBe(
      "unpublished in Label Studio",
    );
  });
});
