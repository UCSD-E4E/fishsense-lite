import { env } from "./env";

export type LabelStudioProject = {
  id: number;
  title: string;
};

export async function getProject(
  id: number,
  revalidate: number,
): Promise<LabelStudioProject> {
  const response = await fetch(`${env.labelStudioUrl}/api/projects/${id}`, {
    headers: { Authorization: `Token ${env.labelStudioApiKey}` },
    next: { revalidate },
  });

  if (!response.ok) {
    throw new Error(
      `Label Studio project ${id} fetch failed: ${response.status} ${response.statusText}`,
    );
  }

  const data = (await response.json()) as { id: number; title: string };
  return { id: data.id, title: data.title };
}

export async function getProjects(
  ids: number[],
  revalidate: number,
): Promise<LabelStudioProject[]> {
  return Promise.all(ids.map((id) => getProject(id, revalidate)));
}
