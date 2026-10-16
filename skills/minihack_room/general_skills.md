### GENERAL SKILLS ###

1. SafeExploratoryMove -> GoalTileDetection -> DirectMoveOntoGoal
Apply when: Only for these task types, with matching prerequisites: room_navigation

2. PerceiveGoalInPartialObservation -> MoveDirectlyOntoAdjacentGoalTile
Apply when: Only for these task types, with matching prerequisites: room_navigation

3. ExploreToRevealMap -> ApproachVisibleDownstairs -> StepOntoDownstairs
Apply when: Only for these task types, with matching prerequisites: room_navigation

4. ScanDungeonView -> LocateStairsRelativeToAgent -> MoveDirectlyTowardGoal
Apply when: Only for these task types, with matching prerequisites: room_navigation

5. ParseDungeonView -> LocateStairsRelativeToPlayer -> MoveDirectlyTowardStairs
Apply when: Only for these task types, with matching prerequisites: room_navigation

6. MoveIntoUnexploredSpace -> RevealMapAndDetectGoalTile -> StepDirectlyOntoGoalTile
Apply when: Only for these task types, with matching prerequisites: room_navigation

7. ObserveDungeonView -> LocateStaircaseRelativeToPlayer -> MoveDirectlyTowardGoalTile
Apply when: Only for these task types, with matching prerequisites: room_navigation

8. ExploratoryMoveToRevealMap -> MoveTowardVisibleDownstairs -> StepOntoDownstairsTile
Apply when: Only for these task types, with matching prerequisites: room_navigation

9. LocateGoalTileRelativeToAgent -> StepTowardGoalAlongSafeFloorTiles -> EnterGoalTile
Apply when: Only for these task types, with matching prerequisites: room_navigation

10. ParseDungeonViewToLocateStairs -> VerifyDirectPathAvoidsHazards -> ExecuteAdjacentMoveOntoStairs
Apply when: Only for these task types, with matching prerequisites: room_navigation

11. ScanPartialDungeonView -> LocateGoalTileRelativeToPlayer -> ExecuteDirectAdjacentMovementOntoGoal
Apply when: Only for these task types, with matching prerequisites: room_navigation

12. ScanPartialObservationForGoalTile -> SelectDirectAdjacentMoveTowardGoal -> ExecuteMoveOntoGoalTile
Apply when: Only for these task types, with matching prerequisites: room_navigation

13. ScanObservationForGoalTile -> StepTowardGoal -> ReorientWhenMoveIsBlockedByWall -> StepOntoGoalTile
Apply when: Only for these task types, with matching prerequisites: room_navigation

14. MoveInUnexploredDirection -> DetectDownstairsTile(>)InNewObservation -> StepDirectlyOntoDownstairsTile
Apply when: Only for these task types, with matching prerequisites: room_navigation

15. PerceiveGoalTileInLocalView -> ComputeDirectRelativeDirectionToGoal -> ExecuteAdjacentMoveOntoStairTile
Apply when: Only for these task types, with matching prerequisites: room_navigation
