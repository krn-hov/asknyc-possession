# Put the match data here

This folder is empty on purpose. The match data is licensed by the organisers of
More than a Hack 2027 and may not be redistributed, so it is not in this repository.

1. Enrol at <https://landing.mobileworldcapital.com/more-than-a-hack-2027> and accept the terms.
   You receive a download link.
2. From the download, copy these three files into `data/tactical/`, keeping their names:

```
data/
└─ tactical/
   ├─ BAR-ATH_tactical_atd.txt               tracking, players and ball (196 MB)
   ├─ BAR-ATH_tactical_atd.xml               tracking metadata
   └─ BAR-ATH_tactical_smart_tagging.xml     automatic tactical tagging
```

Nothing else from the download is used. The pipeline does not need the videos, the
homographies or the broadcast recording.

If you keep the data somewhere else, point `KH_DATA_DIR` at the folder that contains `tactical/`:

```bash
export KH_DATA_DIR=/path/to/folder          # macOS / Linux
$env:KH_DATA_DIR = "D:\path\to\folder"      # Windows PowerShell
```

Everything you put in here is git-ignored (see `../.gitignore`), so it cannot be committed by
accident.
