import React from 'react';
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

const Statistics: React.FC = () => {
  return (
    <div className="flex flex-col items-center w-full">
      <Card className="w-full text-center shadow-xl mb-6">
        <CardHeader>
          <CardTitle className="font-bold text-4xl">Estadísticas del modelo</CardTitle>
        </CardHeader>
      </Card>

      <div className="flex flex-col md:flex-row justify-center gap-6 w-full">
        <Card className="w-full md:w-1/2 shadow-md p-4">
          <CardHeader className='font-bold text-2xl mb-4'>Gráfica de aprendizaje</CardHeader>
          <CardContent className="h-96">
            <img src="/Figure_2.png" alt="Grafica" className="w-full h-full object-contain rounded-lg shadow-2xl" />
          </CardContent>
        </Card>
        <Card className="w-full md:w-1/2 shadow-md p-4">
          <CardHeader className='font-bold text-2xl mb-4'>Matriz de confusión</CardHeader>
          <CardContent className="h-96">
            <img src="/matriz_de_confusion.png" alt="Matriz" className="w-full h-full object-contain rounded-lg shadow-md" />
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default Statistics;