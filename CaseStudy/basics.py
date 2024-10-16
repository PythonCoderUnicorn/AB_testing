
import pandas as pd 

data = pd.DataFrame(
  {
    'Name':['Anna','Karen','John','Alice','Kevin','Janna','Emily','Joana','Alex','Mika'],
    'Age': [35,30,57,44,56,29,22,33,29,37],
    'Salary': [39000,46700,39899,45900,55900,45788,67890,67344,62455,59544],
    'Department': ["Tech","Tech","Tech","Security",'Healthcare','Management','Tech','Security','Healthcare','Management']
  }
)

# print("Sorted by Salary\n", data.sort_values(by='Salary',ascending=True) )
print("Sorted by Salary\n", data.sort_values(by='Salary',ascending=False) )

print("\nGroupby Department & Salary count\n", data.groupby("Department")["Salary"].count()   )
print("\nGroupby Department & Salary average\n", data.groupby("Department")["Salary"].mean()   )
print("\nGroupby Department & Salary min\n", data.groupby("Department")["Salary"].min()   )
print("\nGroupby Department & Age average\n", data.groupby("Department")["Age"].mean()   )



# filtering
print("\nfilter data for Salary \n", data[data["Salary"] > 65e3]  )

condition1 = data[ (data["Salary"] > 45e3 ) & (data["Salary"] < 75e3 )]

print("\nfilter data for Salary >45 & <75 \n", condition1 )

agef = data[data["Age"].isin([30,50])] 
print("\nfilter data for Age .isin(30,50) \n", agef     )



# -- descriptive stats 
print("\nDescriptive Stats\n")
import numpy as np 

numdf = np.random.randint(10,100,size=50)
print(numdf)

print("mean: ", np.mean(numdf))
print("std: ", np.std(numdf))
print("var: ", np.var(numdf))
print("25th quartile: ", np.quantile(numdf,0.25))
print("75th quartile: ", np.quantile(numdf,0.75))
print("50th quartile: ", np.quantile(numdf,0.5))

print( data.describe())



# --- data merging

# left join   (x * () y)
# inner join  ( (*) )
# right join  ( () *)
# left anti join   (*  () )
# right anti join  ( ()  *)

nf = np.random.randint(10,100, size=6)

data1 = pd.DataFrame({
  'key': ['A','B','C','D','E','F','G'],
  'values1': [122,897,677,566,122,677,343]
})
data2 = pd.DataFrame({
  'key': ['C','D','E','F','G','H','J'],
  'values2': [90,78,67,155,190,155,233]
})
print( data1)
print( data2)

print("Merge dataframe")
print("Merge inner\n", pd.merge(data1, data2, on='key', how='inner') )
print("Merge left \n", pd.merge(data1, data2, on='key', how='left') )
print("Merge right \n", pd.merge(data1, data2, on='key', how='right') )
print("Merge left anti \n", pd.merge(data1, data2, on='key', how='left', indicator=True) )










