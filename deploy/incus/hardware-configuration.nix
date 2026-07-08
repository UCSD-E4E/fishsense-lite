# Captured on-box from the running Incus slot (2026-07-07):
#   incus exec fishsense --project fishsense -- nixos-generate-config --show-hardware-config
#
# ⚠️ The root/boot UUIDs below are specific to THIS instance's disk. If the slot
# is ever reprovisioned onto a fresh disk they change and boot breaks — re-capture
# then. The durable long-term fix is a shared golden hardware profile keyed on
# labels (krg-infra ADR 0022 §4) so no per-instance capture is needed; tracked as
# a follow-up.
{
  config,
  lib,
  pkgs,
  modulesPath,
  ...
}: {
  imports = [(modulesPath + "/profiles/qemu-guest.nix")];
  boot.initrd.availableKernelModules = ["ahci" "xhci_pci" "virtio_pci" "virtio_scsi" "sd_mod"];
  boot.initrd.kernelModules = [];
  boot.kernelModules = ["kvm-intel"];
  boot.extraModulePackages = [];
  fileSystems."/" = {
    device = "/dev/disk/by-uuid/f222513b-ded1-49fa-b591-20ce86a2fe7f";
    fsType = "ext4";
  };
  fileSystems."/boot" = {
    device = "/dev/disk/by-uuid/12CE-A600";
    fsType = "vfat";
    options = ["fmask=0022" "dmask=0022"];
  };
  swapDevices = [];
  nixpkgs.hostPlatform = lib.mkDefault "x86_64-linux";
}
