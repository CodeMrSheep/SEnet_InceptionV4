from SE_Inception_V4 import SEInceptionV4
from LoadDataset import LoadDataset
from sklearn.model_selection import train_test_split
import numpy as np

X, y = LoadDataset()

model = SEInceptionV4(include_top=True, input_shape=(256, 256, 3), classes=4)

model.compile(optimizer= 'adam', loss = 'binary_crossentropy', metrics= ['accuracy'])

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0, random_state=42)

history = model.fit(np.array(train_x), np.array(train_y), epochs=25, batch_size=32)

model.save('model.h5')

