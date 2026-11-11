#!/bin/bash
# University of Iowa Musical Instrument Samples, piano (free, "without restrictions").  Every 3rd semitone, ff and mf.
cd "$(dirname "$0")/data/piano"
names=(C Db D Eb E F Gb G Ab A Bb B)
for midi in $(seq 24 3 96); do n=${names[$((midi%12))]}$((midi/12-1))
  for dyn in ff mf; do f="Piano.$dyn.$n.aiff"; [ -s "$f" ] || curl -sS -f -m 300 -o "$f" "https://theremin.music.uiowa.edu/sound%20files/MIS/Piano_Other/piano/$f" || echo "FAILED $f"; done; done
ls | wc -l; du -sh .
