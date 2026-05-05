function required(name: string): string {
  const value = process.env[name];
  if (!value) {
    throw new Error(`Missing required env var: ${name}`);
  }
  return value;
}

const ENV_VARS = {
  fishsenseApiUrl: "FISHSENSE_API_URL",
  fishsenseApiUsername: "FISHSENSE_API_USERNAME",
  fishsenseApiPassword: "FISHSENSE_API_PASSWORD",
  labelStudioUrl: "LABEL_STUDIO_URL",
  labelStudioApiKey: "LABEL_STUDIO_API_KEY",
} as const;

type EnvKey = keyof typeof ENV_VARS;
type Env = Record<EnvKey, string>;

export const env: Env = new Proxy({} as Env, {
  get(_target, prop) {
    if (typeof prop === "string" && prop in ENV_VARS) {
      return required(ENV_VARS[prop as EnvKey]);
    }
    return undefined;
  },
});

export const __test = { required };
