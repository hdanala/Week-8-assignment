import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# Load & Clean 
df = pd.read_csv('healthcare-dataset-stroke-data.csv')
df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
df = df[df['gender'] != 'Other']  


# Stroke Rate by Age Group (Horizontal Bar, Sorted)

bins   = [0, 20, 40, 60, 80, 100]
labels = ['0–20', '21–40', '41–60', '61–80', '80+']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=True)

age_stats = (df.groupby('age_group', observed=True)['stroke']
               .agg(['mean', 'count'])
               .rename(columns={'mean': 'rate', 'count': 'n'}))
age_stats['pct'] = age_stats['rate'] * 100
age_stats = age_stats.sort_values('pct', ascending=True)

fig, ax = plt.subplots(figsize=(9, 4.5))

colors = ['red' if v >= 8 else 'blue' for v in age_stats['pct']]
bars = ax.barh(age_stats.index, age_stats['pct'], height=0.55)

for bar, (_, row) in zip(bars, age_stats.iterrows()):
    ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
            f"{row['pct']:.1f}%  (n={int(row['n'])})",
            va='center', ha='left')

ax.set_xlabel('Stroke Rate (%)')
ax.set_title('Stroke Rate by Age Group')
ax.set_xlim(0, 30)

ax.grid(axis='x', linestyle='--')
ax.grid(axis='y', visible=False)

low_p  = mpatches.Patch(label='Lower risk (<8%)')
high_p = mpatches.Patch(label='Higher risk (≥8%)')
ax.legend(handles=[low_p, high_p], loc='lower right')

plt.tight_layout()
plt.savefig('chart1_age_stroke.png', dpi=150)
plt.show()
print("Chart 1 saved: chart1_age_stroke.png")


# Avg Glucose Level vs Age by Stroke Outcome (Scatter + Trend Lines)
no_stroke  = df[df['stroke'] == 0].sample(500, random_state=42)
yes_stroke = df[df['stroke'] == 1]

fig, ax = plt.subplots(figsize=(9, 4.5))

ax.scatter(no_stroke['age'],  no_stroke['avg_glucose_level'],
           alpha=0.3, s=18, label='No Stroke')

ax.scatter(yes_stroke['age'], yes_stroke['avg_glucose_level'],
           alpha=0.7, s=30, label='Stroke', marker='D')

# Linear trend lines
for subset in [no_stroke, yes_stroke]:
    z = np.polyfit(subset['age'], subset['avg_glucose_level'], 1)
    p = np.poly1d(z)
    x_range = np.linspace(subset['age'].min(), subset['age'].max(), 200)
    ax.plot(x_range, p(x_range), linewidth=2)

# Diabetic threshold
ax.axhline(126, linestyle='--', label='Diabetic threshold (126 mg/dL)')

ax.set_xlabel('Age (years)')
ax.set_ylabel('Avg Glucose Level (mg/dL)')
ax.set_title('Glucose Level vs. Age by Stroke Outcome')

ax.grid(linestyle='--')
ax.legend()

plt.tight_layout()
plt.savefig('chart2_glucose_age.png', dpi=150)
plt.show()
print("Chart 2 saved: chart2_glucose_age.png")


# Heatmap: Stroke Rate by Age Group vs Risk Factor

risk_cols = {
    'Hypertension':         df['hypertension'] == 1,
    'Heart Disease':        df['heart_disease'] == 1,
    'Ever Married':         df['ever_married'] == 'Yes',
    'Smokes':               df['smoking_status'] == 'smokes',
    'Formerly Smoked':      df['smoking_status'] == 'formerly smoked',
    'Never Smoked':         df['smoking_status'] == 'never smoked',
    'Urban':                df['Residence_type'] == 'Urban',
    'Rural':                df['Residence_type'] == 'Rural',
    'High Glucose\n(>126)': df['avg_glucose_level'] > 126,
}

matrix = pd.DataFrame(index=labels, columns=list(risk_cols.keys()), dtype=float)

for age_label in labels:
    age_mask = df['age_group'] == age_label
    for risk_label, risk_mask in risk_cols.items():
        sub = df[age_mask & risk_mask]
        matrix.loc[age_label, risk_label] = (
            sub['stroke'].mean() * 100 if len(sub) > 5 else np.nan
        )

matrix = matrix.astype(float)

fig, ax = plt.subplots(figsize=(12, 5.5))

im = ax.imshow(matrix.values, aspect='auto', vmin=0, vmax=35)

cbar = fig.colorbar(im, ax=ax, pad=0.02)
cbar.set_label('Stroke Rate (%)')

ax.set_xticks(range(len(matrix.columns)))
ax.set_xticklabels(matrix.columns)
ax.set_yticks(range(len(matrix.index)))
ax.set_yticklabels(matrix.index)

for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        val = matrix.values[i, j]
        if not np.isnan(val):
            ax.text(j, i, f'{val:.1f}%', ha='center', va='center')
        else:
            ax.text(j, i, 'N/A', ha='center', va='center')

ax.set_title('Stroke Rate Heatmap: Age Group vs Risk Factor (%)')

plt.tight_layout()
plt.savefig('chart3_heatmap.png', dpi=150)
plt.show()

print("Chart 3 saved: chart3_heatmap.png")


