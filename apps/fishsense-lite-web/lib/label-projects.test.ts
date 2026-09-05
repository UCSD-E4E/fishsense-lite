import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("./fishsense-api", () => ({ getProjectIds: vi.fn() }));

import { getProjectIds } from "./fishsense-api";
import { hasPendingWork, isPublished, liveProjectIds } from "./label-projects";

const idsMock = vi.mocked(getProjectIds);

beforeEach(() => {
  idsMock.mockReset();
  idsMock.mockResolvedValue([7, 3, 9]);
  vi.stubEnv("LABEL_STUDIO_ENABLED", "true");
});

afterEach(() => {
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
});

describe("liveProjectIds", () => {
  it("asks the API for the kind's outstanding projects", async () => {
    expect(await liveProjectIds("laser", 60)).toEqual([7, 3, 9]);
    expect(idsMock).toHaveBeenCalledExactlyOnceWith("laser", 60);
  });

  // The kill switch is the check triage never had: with Label Studio off the
  // landing page went blank while triage carried on calling it. Sharing this
  // function is what makes that impossible to get wrong in one place only.
  it("returns nothing, and asks nothing, when Label Studio is switched off", async () => {
    vi.stubEnv("LABEL_STUDIO_ENABLED", "false");
    expect(await liveProjectIds("laser", 60)).toEqual([]);
    expect(idsMock).not.toHaveBeenCalled();
  });

  // Order belongs to the consumer: the landing page shows cards in API order,
  // triage wants newest dive first. Imposing one here would change the other.
  it("does not reorder", async () => {
    expect(await liveProjectIds("species", 60)).toEqual([7, 3, 9]);
  });

  // A discovery failure must reach the caller. Swallowing it into an empty
  // list reports a drained queue while Label Studio is full of work.
  it("propagates a failure rather than returning empty", async () => {
    idsMock.mockRejectedValue(new Error("503 Service Unavailable"));
    await expect(liveProjectIds("laser", 60)).rejects.toThrow(/503/);
  });
});

describe("isPublished", () => {
  it("keeps a published project", () => {
    expect(isPublished({ id: 1, title: "p", isPublished: true })).toBe(true);
  });

  it("drops an explicitly unpublished one", () => {
    expect(isPublished({ id: 1, title: "p", isPublished: false })).toBe(false);
  });

  // Fails open: only an explicit flag hides a project, so a Label Studio
  // response change cannot silently blank every surface at once.
  it("keeps one whose publish state is unknown", () => {
    expect(isPublished({ id: 1, title: "p" })).toBe(true);
  });
});

describe("hasPendingWork", () => {
  const project = (taskCount?: number, annotatedCount?: number) => ({
    id: 1,
    title: "p",
    isPublished: true,
    taskCount,
    annotatedCount,
  });

  it("keeps a project with unannotated tasks", () => {
    expect(hasPendingWork(project(24, 18))).toBe(true);
  });

  // This is the case that made the landing page list nine projects next to an
  // empty triage queue: our own `completed` column said work remained while
  // Label Studio had every task annotated.
  it("drops a project whose tasks are all annotated", () => {
    expect(hasPendingWork(project(24, 24))).toBe(false);
  });

  it("drops an empty project", () => {
    expect(hasPendingWork(project(0, 0))).toBe(false);
  });

  // Fails OPEN. Hiding real labeling work is the worse error — a labeler
  // cannot act on a queue that never mentions it, and nothing reports the
  // omission.
  it("keeps a project when Label Studio reports no counts", () => {
    expect(hasPendingWork(project(undefined, undefined))).toBe(true);
    expect(hasPendingWork(project(24, undefined))).toBe(true);
    expect(hasPendingWork(project(undefined, 0))).toBe(true);
  });
});
