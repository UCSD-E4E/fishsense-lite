import { env } from "./env";

export type LabelStudioProject = {
  id: number;
  title: string;
};

export async function getProject(
  id: number,
  revalidate: number,
): Promise<LabelStudioProject> {
  const url = `${env.labelStudioUrl}/api/projects/${id}`;
  const response = await fetch(url, {
    headers: { Authorization: `Token ${env.labelStudioApiKey}` },
    next: { revalidate },
  });

  if (!response.ok) {
    console.error(
      `[label-studio] project ${id} fetch failed`,
      { url, status: response.status, statusText: response.statusText },
    );
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
