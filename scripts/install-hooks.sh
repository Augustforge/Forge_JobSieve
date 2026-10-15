#!/usr/bin/env bash
# Install the pre-commit privacy check into this clone.
set -e
root="$(git rev-parse --show-toplevel)"
cat > "$root/.git/hooks/pre-commit" <<'HOOK'
#!/usr/bin/env bash
exec python "$(git rev-parse --show-toplevel)/scripts/privacy_check.py" --staged
HOOK
chmod +x "$root/.git/hooks/pre-commit"
echo "pre-commit privacy check installed"
