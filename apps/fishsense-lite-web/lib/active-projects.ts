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

/** Drop unpublished projects.
 *
 * The id list comes from fishsense-api (`label-studio-project-ids?incomplete=true`),
 * which is derived from label rows and knows nothing about Label Studio's
 * publish state. So a project that is a draft — still being populated — or one
 * deliberately unpublished to hold it back from labelers would still be linked
 * from the landing page. Publish state lives in LS, so it's filtered here,
 * after the per-project fetch that already tells us.
 */
function published(projects: LabelStudioProject[]): LabelStudioProject[] {
  // `!== false` rather than truthiness: only an explicit unpublished flag
  // hides a card. `getProject` already normalizes a missing `is_published`
  // to true, and failing open here too means a Label Studio response change
  // can never silently blank the landing page.
  return projects.filter((project) => project.isPublished !== false);
}

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
  return {
    laser: published(laser),
    species: published(species),
    headtail: published(headtail),
    slate: published(slate),
  };
}
