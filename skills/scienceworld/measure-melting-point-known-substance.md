### TASK: measure-melting-point-known-substance ###

1. TeleportToTaskRoom -> FocusOnMeasuringInstrument -> FocusOnTargetSubstance -> FocusOnAnswerObjectSelectedByMeasuredProperty
Apply when: When solving a measure-melting-point-known-substance task and its observed prerequisites hold.

2. NavigateTo(task_relevant_room) -> FocusOn(measurement_instrument/thermometer) -> FocusOn(target_substance) -> FocusOn(answer_box_matching_comparison_result)
Apply when: When solving a measure-melting-point-known-substance task and its observed prerequisites hold.

3. focus on thermometer -> focus on the target substance -> focus on the box corresponding to the melting point comparison (above-threshold box or below-threshold box, as specified by the task's conditional)
Apply when: When solving a measure-melting-point-known-substance task and its observed prerequisites hold.

4. go_to_room_containing_task_objects -> focus_on_measurement_instrument -> focus_on_target_substance -> focus_on_answer_object_selected_by_comparing_measured_property_to_task_threshold
Apply when: When solving a measure-melting-point-known-substance task and its observed prerequisites hold.

5. TeleportToTaskLocation (kitchen, where the thermometer, substance, and answer boxes reside) -> FocusOnMeasurementInstrument (focus on thermometer) -> FocusOnTargetSubstance (focus on lead) -> EvaluateThresholdCondition (use the substance's known melting point — lead melts far above -10.0 degrees celsius — to select the correct branch) -> FocusOnAnswerObject (focus on the orange box for 'above threshold')
Apply when: When solving a measure-melting-point-known-substance task and its observed prerequisites hold.

6. go to <room named in task goal> -> focus on thermometer -> focus on <target substance> -> focus on <answer box whose color matches the melting-point comparison stated in the goal>
Apply when: When solving a measure-melting-point-known-substance task and its observed prerequisites hold.
