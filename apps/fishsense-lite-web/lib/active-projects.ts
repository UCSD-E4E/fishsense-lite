import { hasPendingWork, isPublished, liveProjectIds } from "./label-projects";
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
  // `liveProjectIds` owns the kill switch, the gate filter and the ordering —
  // the same definition triage uses. See `lib/label-projects.ts`.
  const [laserIds, speciesIds, headtailIds, slateIds] = await Promise.all([
    liveProjectIds("laser", revalidate),
    liveProjectIds("species", revalidate),
    liveProjectIds("headtail", revalidate),
    liveProjectIds("dive-slate", revalidate),
  ]);

  // Never resolve an empty list. With Label Studio switched off every list is
  // empty, and this is what keeps the page from touching it at all.
  const resolve = async (ids: number[]) =>
    ids.length === 0 ? [] : getProjects(ids, revalidate);

  const [laser, species, headtail, slate] = await Promise.all([
    resolve(laserIds),
    resolve(speciesIds),
    resolve(headtailIds),
    resolve(slateIds),
  ]);

  // Published AND still holding work a human can act on — the same two
  // conditions triage applies before walking a project. Linking a project
  // whose tasks are all annotated sends a labeler somewhere with nothing to
  // do, which is what "9 projects" next to an empty triage queue meant.
  const offerable = (projects: LabelStudioProject[]) =>
    projects.filter(isPublished).filter(hasPendingWork);

  return {
    laser: offerable(laser),
    species: offerable(species),
    headtail: offerable(headtail),
    slate: offerable(slate),
  };
}
