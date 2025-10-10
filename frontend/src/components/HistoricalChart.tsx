// frontend/src/components/HistoricalChart.tsx
import React from 'react';
import { Chart as ChartJS, LineElement, CategoryScale, LinearScale, Tooltip, Legend } from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(LineElement, CategoryScale, LinearScale, Tooltip, Legend);

interface Props {
  historical: any[];
}

const HistoricalChart: React.FC<Props> = ({ historical }) => {
  const labels = historical.map(h => h.ts);
  const aqi = historical.map(h => h.aqi);
  const pm25 = historical.map(h => h.pm25);
  // Add more for PM10, NO2, etc.

  return (
    <Line data={{
      labels,
      datasets: [
        { label: 'AQI', data: aqi, borderColor: '#8b5cf6' },
        { label: 'PM2.5', data: pm25, borderColor: '#06b6d4' },
      ]
    }} />
  );
};

export default HistoricalChart;