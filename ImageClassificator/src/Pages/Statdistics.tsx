import React from 'react';
import GraphsVisualization from '@/components/ui/graphsVisualization'

const Statdistics: React.FC = () => {
  return (
    <div className="statistics-container">
      <h1 className="text-2xl font-bold mb-4">Estadísticas del modelo</h1>
      
      <GraphsVisualization type="Matrix" title="Confusion Matrix" />
    </div>
  );
};


export default Statdistics;