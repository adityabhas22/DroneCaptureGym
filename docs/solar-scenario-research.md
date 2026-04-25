# Solar Scenario Research Notes

This pass keeps reward logic unchanged and focuses on richer scenario inputs and tool affordances.

## Field Conditions to Model

- Valid thermal inspections need enough irradiance and stable conditions. Practical guidance commonly uses irradiance above 600 W/m2, low cloud cover, and wind below roughly 14-15 mph.
- Thermal surveys should capture RGB context too. Thermal imagery exposes hotspots, bypass-diode behavior, string outages, PID-like patterns, and soiling/shading heat signatures, while RGB helps explain physical damage, vegetation, soiling, cracked glass, and access constraints.
- Real drone inspections depend on flight planning and data quality. Useful scenario pressure comes from GSD/standoff, camera angle, glare/reflections, overlap, wind blur, no-fly areas, launch/return margin, and route replanning around obstacles.
- False positives are important. Soiling, emissivity changes, glare, shading, and weather changes can make thermal cues ambiguous, so scenarios should reward later inspection/reporting logic only after the reward layer is added.

## Scenario Families

- `single_hotspot`: favorable weather, one clear thermal hotspot, good starter case.
- `soiling_and_shadow`: lower-severity heating plus vegetation shadow, needs RGB explanation.
- `bypass_diode_fault`: thermal cue that should prompt RGB/context evidence.
- `false_positive_glare`: strong irradiance and glare risk, designed to create misleading thermal cues.
- `blocked_corridor_replan`: temporary obstacle blocks direct close-up path, intended to exercise replanning tools.
- `low_battery_tradeoff`: reduced starting battery and higher wind, intended to force return-margin awareness.

## Tooling Lessons Carried Over

- From ClaimsGym: keep deterministic suites centralized; keep model-facing workflow guidance visible-only; keep tool catalogs explicit and machine-readable.
- From OpsArena: compute action availability from current visible state so closed/blocked actions do not look equally valid.

## Sources

- Aerial Accuracy, solar thermal inspection guide: https://aerialaccuracy.com/resources/solar-panel-inspection-guide
- Drone Launch Academy, solar inspection guide: https://dronelaunchacademy.com/resources/drone-solar-panel-inspection-2/
- IEC 62446-3 overview and UAV thermography research reference: https://www.researchgate.net/publication/304624479_OUTDOOR_NON_DESTRUCTIVE_INFRARED_THERMOGRAPHY_OF_PHOTOVOLTAIC_MODULES_AND_PLANTS_FOR_INSPECTION_IEC_62446-3
- UAV RGB/IR PV defect detection research: https://arxiv.org/abs/2111.11709
