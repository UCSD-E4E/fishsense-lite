import { labelStudioEnabled } from "./env";
import { getIncompleteProjectIds } from "./fishsense-api";
import { getProjects, type LabelStudioProject } from "./label-studio";

export type ActiveProjects = {
  laser: LabelStudioProject[];
  species: LabelStudioProject[];
  headtail: LabelStudioProject[];
  slate: LabelStudioProject[];
};

// Fresh object per call — a shared constant would hand every caller the
// same mutable arrays.
const noActiveProjects = (): ActiveProjects => ({
  laser: [],
  species: [],
  headtail: [],
  slate: [],
});

export async function getActiveProjects(revalidate = 300): Promise<ActiveProjects> {
  // Label Studio is off by default — see `labelStudioEnabled`. Short-circuit
  // before the fishsense-api call too: the project IDs it returns are only
  // ever used to resolve names out of Label Studio, so fetching them would
  // be pure waste.
  if (!labelStudioEnabled()) {
    return noActiveProjects();
  }

  const ids = await getIncompleteProjectIds(revalidate);
  const [laser, species, headtail, slate] = await Promise.all([
    getProjects(ids.laser, revalidate),
    getProjects(ids.species, revalidate),
    getProjects(ids.headtail, revalidate),
    getProjects(ids["dive-slate"], revalidate),
  ]);
  return { laser, species, headtail, slate };
}
