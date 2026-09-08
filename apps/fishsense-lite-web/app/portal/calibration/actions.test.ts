import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// Hoisted so the module mocks below can reference them.
const { auth, setCalibrationSource, clearCalibrationSource, revalidatePath } =
  vi.hoisted(() => ({
    auth: vi.fn(),
    setCalibrationSource: vi.fn(),
    clearCalibrationSource: vi.fn(),
    revalidatePath: vi.fn(),
  }));

vi.mock("@/auth", () => ({ auth }));
vi.mock("@/lib/dives", () => ({ setCalibrationSource, clearCalibrationSource }));
vi.mock("next/cache", () => ({ revalidatePath }));

import {
  clearCalibrationSourceAction,
  setCalibrationSourceAction,
} from "./actions";

const ALLOWED_GROUP = "fishsense-portal-admin";
const SIGNED_IN = {
  user: { name: "Alice", email: "a@e.com", groups: [ALLOWED_GROUP] },
};
/** Signed in, but in no group the portal permits. */
const SIGNED_IN_UNAUTHORIZED = {
  user: { name: "Mallory", email: "m@e.com", groups: ["some-other-team"] },
};

beforeEach(() => {
  process.env.PORTAL_ALLOWED_GROUPS = ALLOWED_GROUP;
  auth.mockResolvedValue(SIGNED_IN);
  setCalibrationSource.mockResolvedValue(undefined);
  clearCalibrationSource.mockResolvedValue(undefined);
});

afterEach(() => {
  vi.clearAllMocks();
});

// ── the auth gate ─────────────────────────────────────────────────────
//
// Server actions are ordinary public HTTP endpoints. `/portal/page.tsx`
// redirecting signed-out users does NOT protect them — anyone can POST an
// action directly. If `requireAuthorized` regressed, these would become
// unauthenticated (or merely realm-authenticated) writes to fishsense-api,
// and nothing would fail loudly.
//
// Authentication alone is not enough: the Authentik realm is the whole SSO
// population, so a signed-in stranger must still be refused.

describe("auth gate", () => {
  it("refuses to set a calibration source for a signed-in user in no allowed group", async () => {
    auth.mockResolvedValue(SIGNED_IN_UNAUTHORIZED);

    const result = await setCalibrationSourceAction(9, 5);

    expect(result).toEqual({ ok: false, error: "Not authorized" });
    expect(setCalibrationSource).not.toHaveBeenCalled();
  });

  it("refuses to clear a calibration source for a signed-in user in no allowed group", async () => {
    auth.mockResolvedValue(SIGNED_IN_UNAUTHORIZED);

    const result = await clearCalibrationSourceAction(9);

    expect(result).toEqual({ ok: false, error: "Not authorized" });
    expect(clearCalibrationSource).not.toHaveBeenCalled();
  });

  it("refuses every write when PORTAL_ALLOWED_GROUPS is unset (fails closed)", async () => {
    delete process.env.PORTAL_ALLOWED_GROUPS;

    expect(await setCalibrationSourceAction(9, 5)).toEqual({
      ok: false,
      error: "Not authorized",
    });
    expect(await clearCalibrationSourceAction(9)).toEqual({
      ok: false,
      error: "Not authorized",
    });
    expect(setCalibrationSource).not.toHaveBeenCalled();
    expect(clearCalibrationSource).not.toHaveBeenCalled();
  });

  it("refuses to set a calibration source when there is no session", async () => {
    auth.mockResolvedValue(null);

    const result = await setCalibrationSourceAction(9, 5);

    expect(result).toEqual({ ok: false, error: "Not authenticated" });
    expect(setCalibrationSource).not.toHaveBeenCalled();
  });

  it("refuses to clear a calibration source when there is no session", async () => {
    auth.mockResolvedValue(null);

    const result = await clearCalibrationSourceAction(9);

    expect(result).toEqual({ ok: false, error: "Not authenticated" });
    expect(clearCalibrationSource).not.toHaveBeenCalled();
  });

  it("refuses when a session exists but carries no user", async () => {
    auth.mockResolvedValue({ expires: "later" });

    expect(await setCalibrationSourceAction(9, 5)).toEqual({
      ok: false,
      error: "Not authenticated",
    });
    expect(setCalibrationSource).not.toHaveBeenCalled();
  });

  it("checks the session BEFORE touching the API, not after", async () => {
    // Ordering matters: a check that ran after the write would still return
    // an error while having already mutated prod.
    auth.mockResolvedValue(null);

    await setCalibrationSourceAction(9, 5);
    await clearCalibrationSourceAction(9);

    expect(auth).toHaveBeenCalledTimes(2);
    expect(setCalibrationSource).not.toHaveBeenCalled();
    expect(clearCalibrationSource).not.toHaveBeenCalled();
  });
});

// ── the happy path + guards ───────────────────────────────────────────

describe("setCalibrationSourceAction", () => {
  it("links the dive and revalidates the portal", async () => {
    const result = await setCalibrationSourceAction(9, 5);

    expect(result).toEqual({ ok: true });
    expect(setCalibrationSource).toHaveBeenCalledWith(9, 5);
    expect(revalidatePath).toHaveBeenCalledWith("/portal");
  });

  it("rejects a dive borrowing its own calibration", async () => {
    // Self-reference would make laser-extrinsics resolution loop or 404, and
    // the dive would silently never become measurable.
    const result = await setCalibrationSourceAction(9, 9);

    expect(result).toEqual({
      ok: false,
      error: "A dive cannot be its own calibration source",
    });
    expect(setCalibrationSource).not.toHaveBeenCalled();
  });

  it("returns the failure instead of throwing when the API rejects it", async () => {
    setCalibrationSource.mockRejectedValue(new Error("500 Internal Server Error"));

    const result = await setCalibrationSourceAction(9, 5);

    expect(result).toEqual({ ok: false, error: "500 Internal Server Error" });
    expect(revalidatePath).not.toHaveBeenCalled();
  });

  it("does not revalidate when the write failed", async () => {
    setCalibrationSource.mockRejectedValue(new Error("nope"));

    await setCalibrationSourceAction(9, 5);

    expect(revalidatePath).not.toHaveBeenCalled();
  });
});

describe("clearCalibrationSourceAction", () => {
  it("clears the link and revalidates the portal", async () => {
    const result = await clearCalibrationSourceAction(9);

    expect(result).toEqual({ ok: true });
    expect(clearCalibrationSource).toHaveBeenCalledWith(9);
    expect(revalidatePath).toHaveBeenCalledWith("/portal");
  });

  it("returns the failure instead of throwing when the API rejects it", async () => {
    clearCalibrationSource.mockRejectedValue(new Error("503"));

    expect(await clearCalibrationSourceAction(9)).toEqual({
      ok: false,
      error: "503",
    });
  });

  it("reports a non-Error rejection without crashing", async () => {
    clearCalibrationSource.mockRejectedValue("a bare string");

    expect(await clearCalibrationSourceAction(9)).toEqual({
      ok: false,
      error: "Failed",
    });
  });
});
