# fishsense-lite deploy target — INTERIOR half of the mkTenant contract (ADR 0020).
#
# This is the REPO-ROOT flake that `fishsense-selfupdate` builds on the Incus slot:
# `nixos-rebuild switch --flake github:UCSD-E4E/fishsense-lite#fishsense`. The rest
# of the interior (compose, config, secrets.nix, hardware-configuration.nix) lives
# under `deploy/incus/`; this flake references it by root-relative paths.
{
  description = "fishsense-lite — KRG Incus platform tenant #1";

  inputs = {
    # Pinned to a merged krg-infra main SHA that includes:
    #   #431 vault-agent fishsense.vm cert + repo-owns-deploy runner
    #   #435 ADR 0023 in-slot Temporal client-cert render
    #   #436 quota 6/12   #437 api.fishsense SANs + external_host rename
    #   #438 OIDC secrets → tenant KV + §9 app-secret render pattern
    #   #439 root disk 50 GiB   #440 co-located proxy outpost (Option A)   #443 OIDC writer glob
    # This rev IS our stable contract (ADR 0020 §5); bump deliberately to pick up
    # platform-seam changes.
    krg-infra.url = "github:KastnerRG/krg-infra/2554daa6f026fee17f34827e5bd4e3712dcec3fd?dir=nix";
    nixpkgs.follows = "krg-infra/nixpkgs";
  };

  outputs = {
    self,
    krg-infra,
    nixpkgs,
  }: let
    system = "x86_64-linux";

    tenant = krg-infra.lib.mkTenant {
      name = "fishsense"; # Incus project + OpenBao role + runner scope (provisioned)
      zone = "e4e"; # fronted by the e4e-prod edge (*.e4e.ucsd.edu)
      hostname = "fishsense.e4e.ucsd.edu"; # apex CNAME (published; edge serves a prod LE cert)
      sso.group = "FishSense"; # AD group (per-route auth is in our inner traefik — HANDOFF §7)
      resources = {
        # Bump from the provisioned 4/8 (krg-nat has 16 vCPU / 98 GiB headroom).
        # Boundary quota is terraform-owned: krg-infra PR #436 bumps var.tenants.fishsense
        # + the example flake to match this. ADMIN: `tofu apply` then restart the instance
        # for the RAM raise to take. This block is the reproducible projection.
        cpu = 6;
        ram = "12GiB";
      };
      image = "krg-golden"; # slot boots from the hardened template (already applied)
      compose = ./deploy/incus/compose.yml; # YOUR interior — repo-owns-deploy
      repo = "UCSD-E4E/fishsense-lite"; # LOAD-BEARING: scopes the auto-provisioned runner (ADR 0022)
      # In-VM vault-agent renders a `fishsense-worker` Temporal client cert to
      # /run/tenant/temporal/{tls.crt,tls.key,ca.crt} (ADR 0023 / krg-infra #435).
      # Required — without it no cert renders and the workers can't reach krg-prod Temporal.
      temporal = {namespace = "fishsense";};
    };
  in {
    # The Incus slot (booted at 10.100.0.10) converges to THIS config via our runner.
    # nixosModules.tenant brings the lab baseline (AD-join, firewall, CrowdSec,
    # monitoring) + Docker + the compose runner + the vault-agent cert renders.
    nixosConfigurations.fishsense = nixpkgs.lib.nixosSystem {
      inherit system;
      modules = [
        krg-infra.nixosModules.tenant
        {krg.tenant = tenant;}
        # Incus VM plumbing — the SAME module the krg-golden image builds from
        # (nix/golden): systemd-boot bootloader + ESP/root fileSystems (by label) +
        # incus-agent (keeps `incus exec` working after the switch) + serial console +
        # growPartition. This is why the golden template needs no hardware-configuration.nix;
        # the captured-UUID approach both missed the bootloader and would drop the
        # incus-agent. Robust across reprovision (label-based — ADR 0022 §4).
        ({modulesPath, ...}: {
          imports = [(modulesPath + "/virtualisation/incus-virtual-machine.nix")];
        })
        # Ephemeral VM tier is OEC-exempt — matches krg-golden (nix/golden), which
        # forces this off. base.nix hard-enables OEC (and its module needs an `inputs`
        # specialArg); nixosModules.tenant sets `isVM = true` but does NOT force OEC off,
        # so a tenant converging via its own flake must. (Flagged upstream — this belongs
        # in nixosModules.tenant for isVM tenants.)
        ({lib, ...}: {
          krg.oecQualysTrellix.enable = lib.mkForce false;
        })
        ./deploy/incus/secrets.nix # extends krg.vaultAgent.renders → /run/tenant/secrets/app.env (HANDOFF §9)
      ];
    };

    # Reproducible boundary projection (admin copies into terraform/incus):
    #   nix eval .#krgTenant.terraformTenant --json
    krgTenant = tenant;
  };
}
