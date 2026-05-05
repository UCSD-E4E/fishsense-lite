import { describe, expect, it } from "vitest";
import type { ActiveProjects } from "./active-projects";
import { buildSections } from "./sections";
import type { StaticLink } from "./static-links";

const EMPTY_ACTIVE: ActiveProjects = {
  laser: [],
  species: [],
  headtail: [],
  slate: [],
};

const NO_STATIC: { results: StaticLink[]; admin: StaticLink[] } = {
  results: [],
  admin: [],
};

describe("buildSections", () => {
  it("omits a labeling category when its project list is empty", () => {
    const sections = buildSections(EMPTY_ACTIVE, NO_STATIC);
    expect(sections.map((s) => s.title)).toEqual([]);
  });

  it("turns a single labeling kind into a Labeling-suffixed section", () => {
    const sections = buildSections(
      { ...EMPTY_ACTIVE, laser: [{ id: 42, title: "REEF Laser High" }] },
      NO_STATIC,
    );
    expect(sections).toHaveLength(1);
    expect(sections[0].title).toBe("Laser Labeling");
    expect(sections[0].links).toEqual([
      {
        title: "REEF Laser High",
        description: "REEF Laser High labeling project",
        href: "https://labeler.e4e.ucsd.edu/projects/42",
      },
    ]);
  });

  it("emits sections in the canonical order: Laser, Species, Head/Tail, Slate, Results, Administration", () => {
    const sections = buildSections(
      {
        laser: [{ id: 1, title: "L" }],
        species: [{ id: 2, title: "S" }],
        headtail: [{ id: 3, title: "H" }],
        slate: [{ id: 4, title: "D" }],
      },
      {
        results: [{ title: "Lengths", description: "d", href: "https://r/" }],
        admin: [{ title: "Workflows", description: "w", href: "https://a/" }],
      },
    );
    expect(sections.map((s) => s.title)).toEqual([
      "Laser Labeling",
      "Species Labeling",
      "Head/Tail Labeling",
      "Slate Labeling",
      "Results",
      "Administration",
    ]);
  });

  it("uses Head/Tail (with slash) and Slate as the human labels", () => {
    const sections = buildSections(
      {
        ...EMPTY_ACTIVE,
        headtail: [{ id: 1, title: "H" }],
        slate: [{ id: 2, title: "D" }],
      },
      NO_STATIC,
    );
    expect(sections.map((s) => s.title)).toEqual(["Head/Tail Labeling", "Slate Labeling"]);
  });

  it("preserves project order within a kind", () => {
    const sections = buildSections(
      {
        ...EMPTY_ACTIVE,
        laser: [
          { id: 42, title: "first" },
          { id: 43, title: "second" },
          { id: 44, title: "third" },
        ],
      },
      NO_STATIC,
    );
    expect(sections[0].links.map((l) => l.title)).toEqual(["first", "second", "third"]);
  });

  it("omits Results and Administration when their static link arrays are empty", () => {
    const sections = buildSections(EMPTY_ACTIVE, { results: [], admin: [] });
    expect(sections).toEqual([]);
  });

  it("keeps Results / Administration even when no labeling projects exist", () => {
    const sections = buildSections(EMPTY_ACTIVE, {
      results: [{ title: "Lengths", description: "d", href: "https://r/" }],
      admin: [{ title: "Workflows", description: "w", href: "https://a/" }],
    });
    expect(sections.map((s) => s.title)).toEqual(["Results", "Administration"]);
  });
});
