
"""
A/B testing or split testing 

  is from stats randomized control trials and is used in business

  - test new UX features
  - test new versions of a product
  - test new versions of an algorithm

Control Group       (exposed to one version or the current version)
Experimental Group  (exposed to the new version)


Process:
1. hypothesis of A/B test
2. dest design , Power Analysis , what is needed for stat significance
3. run the A/B test
4. results analysis , stat significance
5. result analysis,  practical significance


Business Hypothesis - what 2 products being compared and what the desired impact / difference
  - how to fix an issue
  - solution will influence the Key Performance Indicators (KPIs)


Primary Metric -
  a way to measure the performance of the product being tested in the A/B test, stat signif
  - single primary metric
  - answering Metric Validity Question ==> higher revenue?   higher engagement?   more views?
  
  Revenue Primary Metric:
    conversion_rate = (number_of_conversions / number_of_visitors_total) * 100
  
  Engagement Primary Metric:
    CTR = (number_of_clicks / number_of_impressions) *100


Hypothesis Testing - determine whether there is a signif diff between observed data ans the expected data
  - test the results of experiment
  - establish stat signif
  
  Null {rejection} Hypothesis (H0):  CTR of 'Learn More' button with Blue is equal to CTR green button
  Alt {accept} Hypothesis (H1):  CTR of 'Learn More' button with green is larger than CTR of blue button

  want to reject the Null


A/B test design 

1. Power Analysis
    
    1.  determine the Power of the test   (1 - beta)
        probability of correctly rejecting the null hypothesis
        beta is Type 2 error
        common to pick 80% as the power
    
    2. determine the significance level of the test
        probability of rejecting the null while the null is true
        detecting sta signif while it's not
        probability of making Type 1 error (alpha)
        common to pick 5% as signif level 
    
    3. determine minimum detectable effect of the test 
        what is the substantive to the stat signif for business?
        proxy that relates to smallest effect that would matter in practice
        no common level, depends on business ask

  beta: probability of Type 2 error
  Power of test: (1-beta)
  alpha: probability of Type 1 error, signif level
  delta: min detectable effect


2. min sample size (avoid p-hacking)
  
  1. primary metric of A/B testing is in binary variable form
  2. primary metric of test is in proportions or averages
  
  H0: avg_ctrl_grp = avg_exp_grp
  H1: avg_ctrl_grp != avg_exp_grp

  xbar_ctrl ~ N(avg_ctrl, std_ctrl^2)
  xbar_exp ~ N(avg_exp, std_exp^2)
  xbar_ctrl - xbar_exp ~ N(avg_ctrl - avg_exp, std_ctrl^2 / N_ctrl + std_exp^2 / N_exp)

  N = (std_ctrl^2 + std_exp^2) * (zeta1 - alpha/2 + zeta1 - beta)^2 / std^2


3. test duration 

  duration = N / num_visitors_per_day
  
  if small test duration then novelty effect, avoid
  if large test duration then maturation effect, users get used it
  

"""





alpha_uppercase = chr(0x0391)
alpha_lowercase = chr(0x03B1)
beta_uppercase = chr(0x0392)
beta_lowercase = chr(0x03B2)
uppercase_delta = chr(0x0394)
lowercase_delta = chr(0x03B4)

# print(uppercase_delta, lowercase_delta)
# print(alpha_uppercase, alpha_lowercase, beta_uppercase, beta_lowercase)











# ----------------------------------------------------------------
#  Control: 'secure free trial!'    vs  Experiment 'Enroll Now'

# 1= clicked 
# 0= no click

import pandas as pd 
import numpy as np 
from scipy.stats import norm
from rich.console import Console 
console = Console()


# read in data 
ab_df = pd.read_csv("https://raw.githubusercontent.com/TatevKaren/CaseStudies/refs/heads/main/AB%20Testing/ab_test_click_data%20(1).csv")


print("\nDataframe Head\n", ab_df.head() )

# describe
desc = ab_df.describe()
console.print("\n[orchid2] Dataframe Description[/orchid2]\n",desc)
print("- -"*20)


# groupby group and sum clicks
group_clicks = ab_df.groupby("group").sum("click")
print("\nGroupby 'group' and sum('click')  \n",group_clicks)
print("- -"*20)

group_counts = ab_df.groupby(['group']).size()
# print( group_counts )

group_click_counts = ab_df.groupby(['group','click']).size().reset_index(name='count')
print("\nGroup Click Counts \n",group_click_counts,'\n')
print("- -"*20)

def column_percent():
  import pandas as pd

  # Sample data
  count_data = {
      'group': ['con', 'con', 'exp', 'exp'],
      'click': [0, 1, 0, 1],
      'count': [8011, 1989, 3884, 6116]
  }

  # Create a DataFrame
  df_count = pd.DataFrame(count_data)

  # Group by 'group' and 'click', then calculate the total count
  grouped_df = df_count.groupby(['group', 'click']).agg({'count': 'sum'}).reset_index()

  # Calculate the total count for each group
  total_count_by_group = grouped_df.groupby('group')['count'].transform('sum')

  # Calculate the percentage of each count within its group
  grouped_df['percentage'] = (grouped_df['count'] / total_count_by_group) * 100

  print(grouped_df)










alpha = 0.05
delta = 0.1
beta = 0 # probability of Type 2 Error 
power = (1 - beta)

console.print(f"\n[gold1]{alpha_lowercase} Alpha {alpha}  (Probability of Type 1 Error | Significance level)")
console.print(f"[hot_pink]{lowercase_delta} Delta {delta}    (min detectable effect)")

# calculate the total number of clicks per group
X_ctrl = ab_df.groupby("group")['click'].sum().loc['con']
X_exp = ab_df.groupby("group")['click'].sum().loc['exp']

groupclick = ab_df.groupby("group")['click'].sum()


console.print(f"\n[honeydew2]Ctrl Group Clicked: {groupclick.iloc[0]}[/]\n[cyan]Exp Group Clicked: {groupclick.iloc[1]}[/]")


# calculating the estimate of click probability per group

N_exp = ab_df[ ab_df['group'] == "con"].count()
N_ctrl = ab_df[ab_df['group'] == "exp"].count()

# p-hat symbol
p_hat =  "\u0070"  + "\u0302" 

p_ctrl_hat = X_ctrl / N_ctrl
p_exp_hat = X_exp / N_exp

print("-"*25)
console.print(f"[plum2]Click Probability {p_hat} Ctrl: ............. {p_ctrl_hat.iloc[0]}")
console.print(f"[plum2]Click Probability {p_hat} Exp: .............. {p_exp_hat.iloc[0]}")

# calculate the estimate of pooled clicked probability
p_pooled_hat = (X_ctrl + X_exp) / (N_ctrl + N_exp)
console.print(f"[grey78]Estimate of pooled clicked probability: {p_pooled_hat.iloc[0]}")
print("-"*25)


# calculate the estimate of pooled variance
pooled_variance = p_pooled_hat * (1 - p_pooled_hat) * (1/N_ctrl + 1/N_exp)
console.print(f"[grey70]{p_hat} pooled: ................ {p_pooled_hat.iloc[0]}")
console.print(f"[grey70]{p_hat} variance: .............. {pooled_variance.iloc[0]}")
print("-"*25)


# calculate the standard error and test stats
standard_error = np.sqrt(pooled_variance)
console.print(f"[orange3]Standard Error: ..........................[/] {standard_error.iloc[0]}")

# calculate the test stat of Z-test
Test_stat = (p_ctrl_hat - p_exp_hat) / standard_error
console.print(f"[yellow4]Test Statistic for 2 sample Z-test: ......[/] {Test_stat.iloc[0]}")


# critical value of Z-test    norm = Normal Distribution  ppf = percent point function
Z_critical = norm.ppf(1 - alpha/2)
console.print(f"[yellow4]Z-critical value from Normal Distribution:[/] {Z_critical}")

# sf = survival function
# p_value = 2 * P[ Z > T]
p_value = 2 * norm.sf(abs(Test_stat))
# 
print("-"*25)

def stat_signif(p_value:float, alpha:float):
  # print rounded p-value 
  console.print(f"[deep_sky_blue2]P-value of the 2-sample Z-test:[/] {p_value[0]}")

  # p_value =< 0.05  then reject null hypothesis
  # p_value >= 0.05  then fail to reject the null hypothesis
  if p_value[0] <= alpha: # .any() 
    console.print("[white] There is [light_green bold]Statistical Significance[/light_green bold] between the groups [/]")
  else:
    console.print("[yellow] There is no Statistical Significance between the groups [/]")



# print("-"*25)
stat_signif(p_value, alpha)



# --------------------------------- data viz
def data_viz(Test_stat, Z_critical):
  import numpy as np
  import pandas as pd
  from scipy.stats import norm
  import matplotlib.pyplot as plt

  # Parameters for the standard normal distribution
  mu = 0  # Mean
  sigma = 1  # Standard deviation
  x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
  y = norm.pdf(x, mu, sigma)


  # Plotting the standard normal distribution
  plt.plot(x, y, label='Standard Normal Distribution')

  # Shade the rejection region for a two-tailed test
  plt.fill_between(x, y, where=(x > Z_critical) | (x < -Z_critical), color='red', alpha=0.5, label='Rejection Region')

  # Adding Test Statistic
  plt.axvline(Test_stat, color='green', linestyle='dashed', linewidth=2, label=f'Test Statistic = {Test_stat}')

  # Adding Z-critical values
  plt.axvline(Z_critical, color='blue', linestyle='dashed', linewidth=1, label=f'Z-critical = {Z_critical}')
  plt.axvline(-Z_critical, color='blue', linestyle='dashed', linewidth=1)

  # Adding labels and title
  plt.xlabel('Z-value')
  plt.ylabel('Probability Density')
  plt.title('Gaussian Distribution with Rejection Region \n (A/B Testing for LunarTech CTA button)')
  plt.legend()

  # Show plot
  plt.show()



Test_stat = Test_stat.iloc[0]
# data_viz(Test_stat.iloc[0], Z_critical)


print("-"*25)
# -------------------------- confidence interval 
# confidence interval for 2-sample Z-test, lower and upper bounds

confidence_interval_lower = round( (p_exp_hat - p_ctrl_hat) - standard_error * Z_critical, 3).iloc[0]
confidence_interval_upper = round( (p_exp_hat - p_ctrl_hat) + standard_error * Z_critical, 3).iloc[0]
diff_CI = confidence_interval_upper - confidence_interval_lower
console.print(f"[pale_turquoise1]Confidence Interval for 2 sample Z-test:[/] ({confidence_interval_lower} , {confidence_interval_upper}) range: {diff_CI :.2f} \n")



# -------------- testing for practical significance in A/B testing
def practical_signif(delta:float, confidence_interval_lower):
  # delta float  min detectable effect
  # 95% confidence interval
  
  lower_bound_CI = confidence_interval_lower
  
  if lower_bound_CI >= delta:
    console.print(f"[spring_green1] There is practical significance with {lowercase_delta} = {delta}[/] \n [grey54]Difference between Ctrl and Experiment Groups[/] \n")
    return True
  else:
    console.print(f"[yellow3] There is no practical significance between Ctrl and Experiment Groups \n")
    return False
  


practical_signif(delta, confidence_interval_lower)

















