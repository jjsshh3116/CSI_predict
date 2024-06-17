import pandas as pd
import matplotlib.pyplot as plt

# CSV 파일 읽기
file_path = 'output/csv/noperson.csv'
data = pd.read_csv(file_path)


# 특정 인덱스의 서브캐리어 선택
selected_indices = [20, 45, 60]
selected_columns = data.columns[selected_indices]

# 선택된 서브캐리어 데이터를 시각화
plt.figure(figsize=(14, 8))
for column in selected_columns:
    plt.plot(data[column], alpha=0.7, label=column)

plt.title('No Person')
plt.xlabel('Time')
plt.ylabel('CSI Amplitude')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()