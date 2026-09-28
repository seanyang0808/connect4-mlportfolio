from dataManagement import dataManager
from modelHandling import model
from pathlib import Path

stage1_data = dataManager.get_data(1, 10)

model = model.CNN()

model.trainModel(
    dataset=stage1_data,
    model_file_path=str(model.get_default_path("testModel_280926")),
    learning_rate=0.0001,
    report_frequency= 500,
    batch_size=16
)

