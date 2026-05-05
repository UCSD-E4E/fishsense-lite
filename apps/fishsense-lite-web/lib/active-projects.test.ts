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
});

afterEach(() => {
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
      ids.map((id) => ({ id, title: `p-${id}` })),
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
        { id: 42, title: "p-42" },
        { id: 43, title: "p-43" },
      ],
      species: [{ id: 70, title: "p-70" }],
      headtail: [{ id: 44, title: "p-44" }],
      slate: [{ id: 66, title: "p-66" }],
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
