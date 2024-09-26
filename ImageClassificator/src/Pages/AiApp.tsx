import React, { useState, useTransition } from 'react';
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

const ImageClassifier = () => {
    const [image, setImage] = useState<File | null>(null);
    const [prediction, setPrediction] = useState<string | null>(null);
    const [confidence, setConfidence] = useState<number | null>(null);
    const [isLoading, setIsLoading] = useState(false);

    const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
        if (event.target.files && event.target.files[0]){
            setImage(event.target.files[0]);
            setPrediction(null);
            setConfidence(null);
        }
    };

    const classifyImage = async () => {
        if(!image) {
            alert('Carge una imagen');
            return;
        }

        setIsLoading(true);
        try{
            const formData = new FormData();
            formData.append('file', image);

            const response = await fetch('http://localhost:5000/classify', {
                method: 'POST',
                body: formData,
            });

            if(!response.ok) {
                throw new Error('Error en la clasificacion')
            }

            const result = await response.json()
            setPrediction(result.class)
            setConfidence(result.confidence)
        } catch(error) {
            alert('Error al clasificar');
        } finally {
            setIsLoading(false);
        }
    };


    return (
        <div className='max-w-md mx-auto mt-10 p-6 bg-white rounded-lg shadow-xl'>
            <h2 className="text-2xl font-bold mb-4">Clasificador de imagenes</h2>
            <div className='grid w-full max-w-sm items-center gap-1.5'>
                <Label htmlFor="picture">Picture</Label>
                <Input id="piture" type="file" />
            </div>
            {image && (
                <img 
                    src={URL.createObjectURL(image)}
                    alt="Preview"
                    className='w-full max-w-xs h-auto object-contain mx-auto'
                    />
            )}
            <br></br>
            <button
                onClick={classifyImage}
                disabled={!image || isLoading}
                className='bg-blue-800 hover:bg-blue-950 text-white font-bold py-2 px-4 rounded disabled:opacity-50'>
                {isLoading ? 'Clasificando...' : 'Clasificar imagen'}
            </button>
            {prediction && (
                <div className='mt-4'>
                    <p><strong>Prediccion:</strong> {prediction}</p>
                </div>
            )}
        </div>
    );
};

export default ImageClassifier