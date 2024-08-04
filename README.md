# PlaWorDet-WARP24
The FIFA Player Worth Determination Model is a sophisticated AI/ML project designed to assess the value of football players based on their statistics and their compatibility with a specific team. Unlike traditional player valuation models, this project considers the existing chemistry of the team and the positional dynamics to provide club-specific player valuations. This enables club owners to make informed bidding decisions in both in-game and real-life FIFA auctions.

## Objective
The primary objective of this project is to determine the worth of football players by integrating various factors:

1. Player Statistics: Analyzing the performance statistics of players.
2. Chemistry with Team: Evaluating how well a player fits into the existing team dynamics.
3. Positional Analysis: Assessing the positional needs of the team and the player's alignment with those needs.

## Features
1. **Club-Specific Valuation**: Calculates the value of a player based on the current players of the club and their statistics.
2. **Bid Recommendation**: Recommends optimal bidding strategies based on the player's worth and team chemistry.
3. **Interactive Interface**: Provides an intuitive interface for users (club owners) to input data and receive valuation insights.
4. **Scalability**: Designed to handle large datasets of player statistics and adaptable to different football leagues and teams.

 ## Usage
1. Input Data: Provide player statistics and team details via the interface.
2. Valuation Process: The model analyzes the input data to determine the player's worth to the team.
3. Recommendations: Receive valuation insights and bidding recommendations based on the analysis.

## Dataset
The project uses a dataset comprising statistics of thousands of active football players across different leagues. The dataset is preprocessed and curated to facilitate accurate player valuations and team chemistry analysis.

## Model Architecture

The FIFA Player Worth Determination Model employs a multi-step process to evaluate player worth:

**Data Preprocessing**: Player feature data is preprocessed to ensure quality and consistency.
**Player Chemistry**: The chemistry of the player with neighboring positions is determined.
**Best Position Analysis**: The best position for the player to play is identified based on team needs.
**Team Attributes Assessmen**t: Overall team attributes are analyzed before and after integrating the new player, comparing the new player's worth with the existing player.
**Regression Model**: The outputs from the DNN models are fed into a few fully connected layers to form a regression model, which determines the player's worth using these features.
