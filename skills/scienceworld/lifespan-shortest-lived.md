### TASK: lifespan-shortest-lived ###

1. TeleportToLocationNamedInTask -> FocusOnTargetObjectMatchingTaskCriterion
Apply when: When solving a lifespan-shortest-lived task and its observed prerequisites hold.

2. look around (survey the specified location) -> look at [candidate object] (examine an animal found there) -> focus on [object matching the task criterion] (focus on the animal satisfying the lifespan condition)
Apply when: When solving a lifespan-shortest-lived task and its observed prerequisites hold.

3. Parse the goal for the target location and selection criterion -> Teleport directly to the specified location ('outside') -> Focus on the object satisfying the criterion (the animal with the shortest life span, e.g., the ant)
Apply when: When solving a lifespan-shortest-lived task and its observed prerequisites hold.

4. teleport to the location named in the goal -> look around to enumerate the candidate objects present in that location -> select the candidate that matches the task criterion by applying domain knowledge about the queried property -> focus on the selected object to submit the answer
Apply when: When solving a lifespan-shortest-lived task and its observed prerequisites hold.

5. go to <task-specified location> -> look around to enumerate candidate objects -> look at each candidate object to confirm its exact in-game name -> focus on the candidate that satisfies the task criterion (chosen via domain knowledge, e.g., the mouse has a shorter lifespan than a beaver)
Apply when: When solving a lifespan-shortest-lived task and its observed prerequisites hold.

6. navigate from starting room to hallway (central hub connecting all rooms) -> teleport to the task-specified location ('outside') when direct 'go to' movement is not admissible -> look around to enumerate the candidate objects (animals) present -> focus on the candidate object matching the task's target property (the animal with the shortest life span), using the object's full in-game name including its descriptor (e.g., 'baby')
Apply when: When solving a lifespan-shortest-lived task and its observed prerequisites hold.
