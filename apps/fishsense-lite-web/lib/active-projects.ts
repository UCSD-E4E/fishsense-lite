import { getIncompleteProjectIds } from "./fishsense-api";
import { getProjects, type LabelStudioProject } from "./label-studio";

export type ActiveProjects = {
  laser: LabelStudioProject[];
  species: LabelStudioProject[];
  headtail: LabelStudioProject[];
  slate: LabelStudioProject[];
};

export async function getActiveProjects(revalidate = 300): Promise<ActiveProjects> {
  const ids = await getIncompleteProjectIds(revalidate);
  const [laser, species, headtail, slate] = await Promise.all([
    getProjects(ids.laser, revalidate),
    getProjects(ids.species, revalidate),
    getProjects(ids.headtail, revalidate),
    getProjects(ids["dive-slate"], revalidate),
  ]);
  return { laser, species, headtail, slate };
}
