import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.audit_room import export_bundle

if __name__ == "__main__":
    print(export_bundle(Path(__file__).resolve().parents[1]))
