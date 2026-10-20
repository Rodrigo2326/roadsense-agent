# Photograph attribution

These photographs are **not** ours. Each is used under its own licence, listed
below, and that licence is separate from the Apache-2.0 covering this repository's
code. Fetched by `scripts/fetch_commons_assets.py`, which refuses any file whose
licence and author it cannot read from the Wikimedia Commons API.

The benchmark does not use these: it runs on the synthetic set, whose degradation
is a measured dial rather than an accident. These are here so the demo shows the
agent working on photographs a person actually took.

| File | Role | Licence | Author | Source |
|---|---|---|---|---|
| `stop-sign.jpg` | a real stop sign | CC0 | Novoklimov | [Commons](https://commons.wikimedia.org/wiki/File%3ARed_octagonal_stop_sign.jpg) |
| `speed-limit.jpg` | a real speed limit sign | CC BY 3.0 | Marek Ślusarczyk (Tupungato) Photo portfolio | [Commons](https://commons.wikimedia.org/wiki/File%3A10_US_Highway_speed_limit_sign_55_MPH_-_Garfield_County%2C_Colorado.jpg) |
| `intersection.jpg` | a real rural intersection | CC0 | Mikepascoe | [Commons](https://commons.wikimedia.org/wiki/File%3AIntersection_of_119th_Street_and_Jasper_Road_at_Canfield%2C_unincorporated_Boulder_County%2C_Colorado%2C_USA_-_2025.jpg) |
| `street-traffic.jpg` | a real street with vehicles (text redacted: 17 regions blurred) | CC0 | Bernard Spragg. NZ from Christchurch, New Zealand | [Commons](https://commons.wikimedia.org/wiki/File%3AStreet_scene_Causeway_Bay_HK._%2816065463789%29.jpg) |

## What was changed, and what was checked

`street-traffic.jpg` is a photograph of real traffic, so it carried real number
plates: Rekognition read `MP 7866` and `GF 4205` off it on the first try. Every
text region it could find has been blurred, and the script re-reads the file
afterwards and fails if any word is still detectable. The copy published here is
therefore strictly less identifying than the original, which remains available
under its own licence at the source link.

The other three carry no plates and no people. Faces are not resolvable in any of
these at the published resolution.
