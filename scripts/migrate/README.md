This folder contains utilities to migrate experiment folders after removing fusion profiles A/B/C.

remove_and_rename_fusion_profiles.sh

- Dry-run by default: lists candidate B/C folders (to delete) and A folders (to rename by removing "_A_").
- Pass `--apply` to actually perform the deletions/renames. Use with care.

Examples:

# Dry-run

./scripts/migrate/remove_and_rename_fusion_profiles.sh

# Apply changes (destructive)

./scripts/migrate/remove_and_rename_fusion_profiles.sh --apply
