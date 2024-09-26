import React, { useState, useEffect } from  'react'

type visualization = 'Graph' | 'Matrix';

interface visualizationProps {
    type: visualization;
    title: string;
}

const graphsVisualization: React.FC<visualizationProps> = ({ type, title}) => {
    const [imgUrl, setImgUrl] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const fetchImage = async() => { 
            setIsLoading(true);
            try{
                const response = await fetch(`http://localhost:5000/${type}`);
                if(!response.ok) {
                    throw new Error('Error al cargar la imagen');
                }
                const blob = await response.blob();
                const url = URL.createObjectURL(blob);
                setImgUrl(url);
                setError(null);
            } catch(e) {
                setError('Error al cargar la imagen');
                setImgUrl(null);
            } finally {
                setIsLoading(false);
            }
        };

        fetchImage();

        return () => {
            if(imgUrl) {
                URL.revokeObjectURL(imgUrl);
            }
        };

    }, [type]);

    if(isLoading){
        return <div>Loading...</div>
    }

    if(error){
        return <div>Error: {error}</div>
    }

    return(
        <div className='Stadistics'>
            <h2>{title}</h2>
            {imgUrl && <img src={imgUrl} alt={title} />}
        </div>
    );
};

export default graphsVisualization;
