import matplotlib.pyplot as plt
import numpy as np

f = {'+': (0.011593448040518697, ('+', 'RHYTHM', 'Rhythm change')),
 'N': (1.996329299108924, ('N', 'NORMAL', 'Normal beat')),
 'A': (0.023124222304974525, ('A', 'APC', 'Atrial premature contraction')),
 'V': (0.06757205000142157, ('V', 'PVC', 'Premature ventricular contraction')),
 '~': (0.005498478099811659, ('~', 'NOISE', 'Signal quality change')),
 '|': (0.0011731769097453673, ('|', 'ARFCT', 'Isolated QRS-like artifact')),
 'Q': (0.00029303638979167777, ('Q', 'UNKNOWN', 'Unclassifiable beat')),
 '/': (0.06654105795358789, ('/', 'PACE', 'Paced beat')),
 'f': (0.008794161106882192, ('f', 'PFUS', 'Fusion of paced and normal beat')),
 'x': (0.001716257314101766,
  ('x', 'NAPC', 'Non-conducted P-wave (blocked APB)')),
 'F': (0.007179643074282036,
  ('F', 'FUSION', 'Fusion of ventricular and normal beat')),
 'j': (0.0020370403316194914, ('j', 'NESC', 'Nodal (junctional) escape beat')),
 'L': (0.07721952339058256, ('L', 'LBBB', 'Left bundle branch block beat')),
 'a': (0.0013333688898370622,
  ('a', 'ABERR', 'Aberrated atrial premature beat')),
 'J': (0.0007373583028321666,
  ('J', 'NPC', 'Nodal (junctional) premature beat')),
 'R': (0.06887880973165825, ('R', 'RBBB', 'Right bundle branch block beat')),
 '[': (5.326657256238847e-05,
  ('[', 'VFON', 'Start of ventricular flutter/fibrillation')),
 '!': (0.004207711165589481, ('!', 'FLWAV', 'Ventricular flutter wave')),
 ']': (5.326657256238847e-05,
  (']', 'VFOFF', 'End of ventricular flutter/fibrillation')),
 'E': (0.000941878959668032, ('E', 'VESC', 'Ventricular escape beat')),
 'S': (1.7754893692574015e-05,
  ('S', 'SVPB', 'Premature or ectopic supraventricular beat')),
 '"': (0.0038944835576151856, ('"', 'NOTE', 'Comment annotation')),
 'e': (0.00014205680496488533, ('e', 'AESC', 'Atrial escape beat'))}

v = {'+': 1291,  'N': 75052,  'A': 2546,  'V': 7130,  '~': 616,  '|': 132,  'Q': 33,  '/': 7028,  'f': 982,  'x': 193,  'F': 803,  'j': 229,  'L': 8075,  'a': 150,  'J': 83,  'R': 7259,  '[': 6,  '!': 472,  ']': 6,  'E': 106,  'S': 2,  '"': 437,  'e': 16}

labels = []
data = []

for k,e in v.items():
    labels.append(k)
    data.append(e)
    
print(data)
# Calculate label distances based on data values
total = sum(data)
label_distances = [1.1 + (i / total) for i in np.cumsum(data)]
label_distances = list(map(float, label_distances))  # convert to float

# Set up the figure
fig, ax = plt.subplots(figsize=(6, 6))

# Create a pie chart with a circle at the center
circle = plt.Circle((0, 0), 0.5, color='white')
ax.add_artist(circle)

# Create the pie chart with dynamic label distances
pie = ax.pie(data, labels=labels, startangle=90, labeldistance=label_distances)

# Set the aspect ratio to be equal so the chart is circular
ax.axis('equal')

# Show the plot
plt.show()