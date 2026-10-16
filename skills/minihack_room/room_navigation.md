### TASK: room_navigation ###

1. SafeExploratoryMove -> GoalTileDetection -> DirectMoveOntoGoal
Apply when: When solving a room_navigation task and its observed prerequisites hold.

2. PerceiveGoalInPartialObservation -> MoveDirectlyOntoAdjacentGoalTile
Apply when: When solving a room_navigation task and its observed prerequisites hold.

3. ExploreToRevealMap -> ApproachVisibleDownstairs -> StepOntoDownstairs
Apply when: When solving a room_navigation task and its observed prerequisites hold.

4. ScanDungeonView -> LocateStairsRelativeToAgent -> MoveDirectlyTowardGoal
Apply when: When solving a room_navigation task and its observed prerequisites hold.

5. ParseDungeonView -> LocateStairsRelativeToPlayer -> MoveDirectlyTowardStairs
Apply when: When solving a room_navigation task and its observed prerequisites hold.

6. MoveIntoUnexploredSpace -> RevealMapAndDetectGoalTile -> StepDirectlyOntoGoalTile
Apply when: When solving a room_navigation task and its observed prerequisites hold.
