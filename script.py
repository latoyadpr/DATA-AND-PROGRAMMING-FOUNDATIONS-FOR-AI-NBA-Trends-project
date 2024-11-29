import numpy as np
import pandas as pd
from scipy.stats import pearsonr, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns

import codecademylib3
np.set_printoptions(suppress=True, precision=2)

nba = pd.read_csv('./nba_games.csv')

# Subset Data to 2010 Season, 2014 Season
nba_2010 = nba[nba.year_id == 2010]
nba_2014 = nba[nba.year_id == 2014]

print(nba_2010.head())
print(nba_2014.head())

# Define knicks_pts_10 and nets_pts_10
knicks_pts_10 = nba_2010[nba_2010.fran_id == 'Knicks'].pts
nets_pts_10 = nba_2010[nba_2010.fran_id == 'Nets'].pts

# Calculate mean scores
knicks_mean_score = np.mean(knicks_pts_10)
nets_mean_score = np.mean(nets_pts_10)

# Calculate the difference in means
diff_means_2010 = knicks_mean_score - nets_mean_score

plt.hist(knicks_pts_10, alpha=0.8, weights=np.ones_like(knicks_pts_10) / len(knicks_pts_10), label='Knicks')
plt.hist(nets_pts_10, alpha=0.8, weights=np.ones_like(nets_pts_10) / len(nets_pts_10), label='Nets')
plt.legend()
plt.title("2010 Season: Knicks vs Nets Points Distribution")
plt.xlabel("Points")
plt.ylabel("Density")
plt.show()


# Define knicks_pts_14 and nets_pts_14
knicks_pts_14 = nba_2014[nba_2014.fran_id == 'Knicks'].pts
nets_pts_14 = nba_2014[nba_2014.fran_id == 'Nets'].pts

# Calculate mean scores for 2014
knicks_mean_score_14 = np.mean(knicks_pts_14)
nets_mean_score_14 = np.mean(nets_pts_14)

# Calculate the difference in means for 2014
diff_means_2014 = knicks_mean_score_14 - nets_mean_score_14
print("Difference in means for 2014:", diff_means_2014)

# Plot overlapping histograms for 2014
plt.clf()  # Clear previous plots
plt.hist(knicks_pts_14, alpha=0.8, weights=np.ones_like(knicks_pts_14) / len(knicks_pts_14), label='Knicks')
plt.hist(nets_pts_14, alpha=0.8, weights=np.ones_like(nets_pts_14) / len(nets_pts_14), label='Nets')
plt.legend()
plt.title("2014 Season: Knicks vs Nets Points Distribution")
plt.xlabel("Points")
plt.ylabel("Density")
plt.show()

plt.clf()  # Clear previous plots
sns.boxplot(data=nba_2010, x='fran_id', y='pts')
plt.title("2010 Season: Points Scored by Franchise")
plt.xlabel("Franchise")
plt.ylabel("Points")
plt.show()


# Calculate the contingency table
location_result_freq = pd.crosstab(nba_2010['game_result'], nba_2010['game_location'])

# Print the result
print(location_result_freq)


# Calculate the table of proportions
location_result_proportions = location_result_freq / len(nba_2010)

# Print the result
print(location_result_proportions)

# Calculate the Chi-Square statistic and expected frequencies
chi2, pval, dof, expected = chi2_contingency(location_result_freq)

# Print the expected table and Chi-Square statistic
print("Expected Table:")
print(expected)
print("Chi-Square Statistic:", chi2)


# Calculate the covariance matrix
cov_matrix = np.cov(nba_2010['forecast'], nba_2010['point_diff'])

# Extract the covariance between forecast and point_diff
point_diff_forecast_cov = cov_matrix[0, 1]

# Print the result
print("Covariance between forecast and point_diff:", point_diff_forecast_cov)


# Calculate the correlation
point_diff_forecast_corr, _ = pearsonr(nba_2010['forecast'], nba_2010['point_diff'])

# Print the result
print("Correlation between forecast and point_diff:", point_diff_forecast_corr)


plt.clf()  # Clear previous plots
plt.scatter(nba_2010['forecast'], nba_2010['point_diff'])
plt.xlabel('Forecasted Win Probability')
plt.ylabel('Point Differential')
plt.title('Scatter Plot of Forecast vs. Point Differential')
plt.show()
