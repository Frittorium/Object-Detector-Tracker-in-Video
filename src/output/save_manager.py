import os
import shutil
from pathlib import Path

import config
from errors import SaveError


class SaveManager:
    def save(self, temporary_path, destination, overwrite=False):
        temporary_path, dest = Path(temporary_path), Path(destination)
        if not temporary_path.is_file():
            raise SaveError("Temporary output missing", "The processed video is no longer available.")
        if dest.suffix.lower() != config.OUTPUT_EXTENSION:
            dest = dest.with_suffix(config.OUTPUT_EXTENSION)
        if not dest.parent.is_dir():
            raise SaveError("Destination folder does not exist.")
        if dest.exists() and not overwrite:
            raise SaveError("A file with that name already exists.")
        size = temporary_path.stat().st_size
        if shutil.disk_usage(dest.parent).free < size * 1.05:
            raise SaveError("Not enough free disk space at the destination.")
        part = dest.with_name(dest.name + ".part")
        try:
            shutil.copy2(temporary_path, part)
            os.replace(part, dest)
        except OSError as e:
            part.unlink(missing_ok=True)
            raise SaveError(f"Save failed: {e}", "The video could not be saved. Please try again.") from e
        if not dest.is_file() or dest.stat().st_size != size:
            raise SaveError("Saved file failed verification.")
        return dest