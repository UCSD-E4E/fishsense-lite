# fishsense-lite deploy target — INTERIOR half of the mkTenant contract (ADR 0020).
#
# This is the REPO-ROOT flake that `fishsense-selfupdate` builds on the Incus slot:
# `nixos-rebuild switch --flake github:UCSD-E4E/fishsense-lite#fishsense`. The rest
# of the interior (compose, config, secrets.nix, hardware-configuration.nix) lives
# under `deploy/incus/`; this flake references it by root-relative paths.
{
  description = "fishsense-lite — KRG Incus platform tenant #1";

  inputs = {
    # Tracks krg-infra `main`; the EXACT rev is pinned in `flake.lock`, not here.
    # That lock rev is our stable contract (ADR 0020 §5): every converge builds from
    # the committed lock, so `main` moving doesn't touch the slot until the lock is
    # advanced — the deliberate act being a merge to our `main`.
    #
    # Advancing the pin is "Axis B" (krg-infra docs/tenant-updates.md) and it's OURS:
    # `nixpkgs.follows = "krg-infra/nixpkgs"`, so bumping krg-infra drags in new
    # nixpkgs (kernel / bash / openssl / CVE fixes). `.github/workflows/update-flake.yml`
    # does it weekly — `nix flake update krg-infra`, committed straight to `main` — and
    # the nightly `system.autoUpgrade` (#460) rolls it out. Skip it and the slot freezes
    # on old, unpatched packages. A new kernel needs a manual `incus restart` (allowReboot=false).
    #
    # The lock currently carries krg-infra @ 4c10ed3e (incl. #435–#440/#443/#453,
    # #459 compose force-recreate, #460 nightly auto-upgrade). `git log -p flake.lock`
    # is the real history of what shipped.
    krg-infra.url = "github:KastnerRG/krg-infra?dir=nix";
    nixpkgs.follows = "krg-infra/nixpkgs";
  };

  outputs = {
    self,
    krg-infra,
    nixpkgs,
  }: let
    system = "x86_64-linux";
    pkgs = nixpkgs.legacyPackages.${system};

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
        ./deploy/incus/workdir.nix # populate /var/lib/krg/fishsense so the compose's relative binds resolve
      ];
    };

    # Reproducible boundary projection (admin copies into terraform/incus):
    #   nix eval .#krgTenant.terraformTenant --json
    krgTenant = tenant;

    # Dev shell (`nix develop`, or auto via direnv + .envrc). Cluster tooling
    # for the NRP/Nautilus data-worker: `kubectl`, the OIDC `kubelogin`
    # (int128/kubelogin → `kubectl oidc-login`, which NRP's CILogon kubeconfig
    # requires — NOT the Azure `kubelogin`), and `kustomize` for
    # `deploy/k8s/data-worker`. Pinned to the same nixpkgs the slot builds from
    # (nixpkgs.follows = krg-infra/nixpkgs), so it advances with the weekly
    # flake bump.
    devShells.${system}.default = pkgs.mkShell {
      packages = [
        pkgs.kubectl
        pkgs.kubelogin-oidc
        pkgs.kustomize
      ];
      shellHook = ''
        echo "fishsense dev shell: kubectl + kubelogin-oidc (kubectl oidc-login) + kustomize"
        echo "NRP login: kubectl get pods  (opens a CILogon browser window the first time)"
      '';
    };
  };
}
