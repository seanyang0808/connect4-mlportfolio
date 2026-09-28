def get_result(dataset: tuple, game_outcome):
    player_that_placed = int(dataset[2][0][0])
    # a version that uses one hot 1d matrices rather than class indices of classification
    # if game_outcome == player_that_placed:
    #     probability_win = 1
    #     probability_loss = 0
    #     probability_draw = 0
    # elif game_outcome == 2:
    #     probability_win = 0
    #     probability_loss = 0
    #     probability_draw = 1
    # else:
    #     probability_win = 0
    #     probability_loss = 1
    #     probability_draw = 0
    # return [probability_win, probability_loss, probability_draw]

    if game_outcome == player_that_placed:
        return 0       # win
    elif game_outcome == 2:
        return 2       # draw
    else:
        return 1       # loss
