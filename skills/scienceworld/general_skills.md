### GENERAL SKILLS ###

1. LookAround -> FocusOnTargetObject
Apply when: Only for these task types, with matching prerequisites: grow-fruit

2. TeleportToLocationNamedInTask -> FocusOnTargetObjectMatchingTaskCriterion
Apply when: Only for these task types, with matching prerequisites: lifespan-shortest-lived

3. look around -> go to workshop -> focus on the inclined plane with the most friction
Apply when: Only for these task types, with matching prerequisites: inclined-plane-friction-named-surfaces

4. TeleportToTaskRoom -> FocusOnMeasuringInstrument -> FocusOnTargetSubstance -> FocusOnAnswerObjectSelectedByMeasuredProperty
Apply when: Only for these task types, with matching prerequisites: measure-melting-point-known-substance

5. NavigateTo(task_relevant_room) -> FocusOn(measurement_instrument/thermometer) -> FocusOn(target_substance) -> FocusOn(answer_box_matching_comparison_result)
Apply when: Only for these task types, with matching prerequisites: measure-melting-point-known-substance

6. select a task-qualifying object already visible in the current room's initial observation (e.g., a clearly non-living object) -> focus on <that object> -> move <that object> to <the specified destination container in the same room>
Apply when: Only for these task types, with matching prerequisites: find-non-living-thing

7. FocusOnTargetObject -> PickUpTargetObject -> TeleportToDestinationRoom -> MoveObjectToTargetContainer -> FocusOnTargetObject (final confirmation to trigger task completion)
Apply when: Only for these task types, with matching prerequisites: find-non-living-thing

8. LookAround -> LookAt(target door) -> GoTo(greenhouse) -> FocusOn(the box matching the dominance determination for this instance)
Apply when: Only for these task types, with matching prerequisites: mendelian-genetics-unknown-plant

9. look around (survey the specified location) -> look at [candidate object] (examine an animal found there) -> focus on [object matching the task criterion] (focus on the animal satisfying the lifespan condition)
Apply when: Only for these task types, with matching prerequisites: lifespan-shortest-lived

10. focus on thermometer (measuring instrument) -> focus on target substance (lead) -> focus on the answer box selected by comparing the substance's melting point against the stated reference temperature (above the threshold -> one box, below -> the other box)
Apply when: Only for these task types, with matching prerequisites: measure-melting-point-known-substance

11. go to greenhouse (location containing the answer boxes) -> focus on the box that matches the determined inheritance conclusion for the unknown plant's trait (green box if the trait is dominant, blue box if the trait is recessive)
Apply when: Only for these task types, with matching prerequisites: mendelian-genetics-unknown-plant

12. look at each candidate inclined plane (both unnamed surfaces) -> activate the stopwatch -> focus on the inclined plane concluded to have the least friction (the terminal answer action that scores the task)
Apply when: Only for these task types, with matching prerequisites: inclined-plane-friction-unnamed-surfaces

13. navigate to the task-specified location (routing through a connected intermediate room when no direct door exists from the current room) -> look around to enumerate the candidate objects present -> focus on the single object that satisfies the task criterion, selecting it via domain knowledge when simple inspection returns no distinguishing information
Apply when: Only for these task types, with matching prerequisites: lifespan-longest-lived

14. look at each named candidate object -> look at the remaining candidate object -> focus on the candidate that answers the task question (here, the inclined plane with the most friction; the correct surface is instance-specific and must not be assumed as a universal material property)
Apply when: Only for these task types, with matching prerequisites: inclined-plane-friction-named-surfaces

15. Parse the goal for the target location and selection criterion -> Teleport directly to the specified location ('outside') -> Focus on the object satisfying the criterion (the animal with the shortest life span, e.g., the ant)
Apply when: Only for these task types, with matching prerequisites: lifespan-shortest-lived
