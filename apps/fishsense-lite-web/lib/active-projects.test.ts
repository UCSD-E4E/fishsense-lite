import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("./fishsense-api", () => ({
  getIncompleteProjectIds: vi.fn(),
}));
vi.mock("./label-studio", () => ({
  getProjects: vi.fn(),
}));

import { getActiveProjects } from "./active-projects";
import { getIncompleteProjectIds } from "./fishsense-api";
import { getProjects } from "./label-studio";

const idsMock = vi.mocked(getIncompleteProjectIds);
const projectsMock = vi.mocked(getProjects);

beforeEach(() => {
  idsMock.mockReset();
  projectsMock.mockReset();
  // The Label Studio path is off by default (see labelStudioEnabled).
  // These tests cover the enabled path, so opt in explicitly.
  vi.stubEnv("LABEL_STUDIO_ENABLED", "true");
});

afterEach(() => {
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
});

describe("getActiveProjects", () => {
  it("resolves IDs once, then resolves names per kind, returning a four-bucket map", async () => {
    idsMock.mockResolvedValue({
      laser: [42, 43],
      species: [70],
      headtail: [44],
      "dive-slate": [66],
    });
    projectsMock.mockImplementation(async (ids) =>
      ids.map((id) => ({ id, title: `p-${id}`, isPublished: true })),
    );

    const result = await getActiveProjects(60);

    expect(idsMock).toHaveBeenCalledExactlyOnceWith(60);
    expect(projectsMock).toHaveBeenCalledTimes(4);
    expect(projectsMock).toHaveBeenNthCalledWith(1, [42, 43], 60);
    expect(projectsMock).toHaveBeenNthCalledWith(2, [70], 60);
    expect(projectsMock).toHaveBeenNthCalledWith(3, [44], 60);
    expect(projectsMock).toHaveBeenNthCalledWith(4, [66], 60);

    expect(result).toEqual({
      laser: [
        { id: 42, title: "p-42", isPublished: true },
        { id: 43, title: "p-43", isPublished: true },
      ],
      species: [{ id: 70, title: "p-70", isPublished: true }],
      headtail: [{ id: 44, title: "p-44", isPublished: true }],
      slate: [{ id: 66, title: "p-66", isPublished: true }],
    });
  });

  it("defaults revalidate to 300 seconds", async () => {
    idsMock.mockResolvedValue({ laser: [], species: [], headtail: [], "dive-slate": [] });
    projectsMock.mockResolvedValue([]);

    await getActiveProjects();

    expect(idsMock).toHaveBeenCalledExactlyOnceWith(300);
    for (const call of projectsMock.mock.calls) {
      expect(call[1]).toBe(300);
    }
  });

  it("returns empty arrays for kinds with no incomplete projects", async () => {
    idsMock.mockResolvedValue({ laser: [], species: [], headtail: [], "dive-slate": [] });
    projectsMock.mockResolvedValue([]);

    const result = await getActiveProjects(60);

    expect(result).toEqual({ laser: [], species: [], headtail: [], slate: [] });
  });
});

// Kill-switch behavior. With LABEL_STUDIO_ENABLED off (the default), the
// landing page must render without touching Label Studio at all: a single
// unresolvable project ID used to throw out of SSR and 500 the whole page.
// buildSections() drops any kind with zero projects, so empty buckets
// collapse the four labeling sections and leave Results + Administration.
describe("getActiveProjects (Label Studio disabled)", () => {
  beforeEach(() => {
    vi.stubEnv("LABEL_STUDIO_ENABLED", "false");
  });

  it("returns an empty four-bucket map", async () => {
    const result = await getActiveProjects(60);

    expect(result).toEqual({ laser: [], species: [], headtail: [], slate: [] });
  });

  it("does not call Label Studio", async () => {
    await getActiveProjects(60);

    expect(projectsMock).not.toHaveBeenCalled();
  });

  it("does not even fetch project IDs from fishsense-api", async () => {
    await getActiveProjects(60);

    expect(idsMock).not.toHaveBeenCalled();
  });

  it("resolves rather than throwing when Label Studio would 401", async () => {
    // Guards the actual prod failure: getProject threw on a 401 and the
    // rejection propagated through Promise.all out of the server component.
    projectsMock.mockRejectedValue(new Error("401 Unauthorized"));

    await expect(getActiveProjects(60)).resolves.toEqual({
      laser: [],
      species: [],
      headtail: [],
      slate: [],
    });
  });

  it("is disabled when LABEL_STUDIO_ENABLED is unset entirely", async () => {
    vi.unstubAllEnvs();

    await getActiveProjects(60);

    expect(idsMock).not.toHaveBeenCalled();
    expect(projectsMock).not.toHaveBeenCalled();
  });
});

describe("getActiveProjects — publish filtering", () => {
  it("hides unpublished projects from the landing page", async () => {
    // The real case: laser project 274728 was unpublished in Label Studio to
    // hold it back from labelers, but the id list comes from fishsense-api
    // (derived from label rows) and knows nothing about publish state — so
    // without this filter the held project stays linked from the landing page.
    idsMock.mockResolvedValue({
      laser: [274728, 73],
      species: [],
      headtail: [],
      "dive-slate": [],
    });
    projectsMock.mockImplementation(async (ids) =>
      ids.map((id) => ({
        id,
        title: `p-${id}`,
        isPublished: id !== 274728,
      })),
    );

    const result = await getActiveProjects(60);

    expect(result.laser).toEqual([{ id: 73, title: "p-73", isPublished: true }]);
  });

  it("filters every kind, not just laser", async () => {
    idsMock.mockResolvedValue({
      laser: [1],
      species: [2],
      headtail: [3],
      "dive-slate": [4],
    });
    projectsMock.mockImplementation(async (ids) =>
      ids.map((id) => ({ id, title: `p-${id}`, isPublished: false })),
    );

    const result = await getActiveProjects(60);

    expect(result).toEqual({ laser: [], species: [], headtail: [], slate: [] });
  });

  it("keeps drafts out while a project is still being populated", async () => {
    // Per-dive projects are created as drafts and only published once their
    // task set is complete, so a half-filled project must not appear either.
    idsMock.mockResolvedValue({
      laser: [],
      species: [100, 101],
      headtail: [],
      "dive-slate": [],
    });
    projectsMock.mockImplementation(async (ids) =>
      ids.map((id) => ({ id, title: `p-${id}`, isPublished: id === 100 })),
    );

    const result = await getActiveProjects(60);

    expect(result.species.map((p) => p.id)).toEqual([100]);
  });
});
