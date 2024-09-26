import React from 'react';
import GraphsVisualization from '@/components/ui/graphsVisualization'

const Statdistics: React.FC = () => {
  return (
    <div className="statistics-container">
      <h1 className="text-2xl font-bold mb-4">Estadísticas del modelo</h1>
      <GraphsVisualization type="Graph" title="Gráfica de aprendizaje" />
      <GraphsVisualization type="Matrix" title="Matriz de confusión" />
    </div>
  );
};


export default Statdistics;