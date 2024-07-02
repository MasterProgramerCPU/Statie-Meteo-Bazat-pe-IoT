document.addEventListener("DOMContentLoaded", function() {
    // Fetch data from the API endpoint
    Promise.all([
        fetch('/api/masuratoare/').then(response => response.json()),
        fetch('/api/temperatura/').then(response => response.json()),
        fetch('/api/presiune/').then(response => response.json()),
        fetch('/api/umiditate/').then(response => response.json()),
        fetch('/api/lumina/').then(response => response.json()),
        fetch('/api/ceata/').then(response => response.json()),
        fetch('/api/anemometru/').then(response => response.json())
    ]).then(dataArrays => {
        const measurements = dataArrays[0];
        const latestMeasurement = measurements.slice(-1)[0];

        const latestTemperature = dataArrays[1].slice(-1)[0];
        const latestPressure = dataArrays[2].slice(-1)[0];
        const latestHumidity = dataArrays[3].slice(-1)[0];
        const latestLight = dataArrays[4].slice(-1)[0];
        const latestFog = dataArrays[5].slice(-1)[0];
        const latestWind = dataArrays[6].slice(-1)[0];

        const liveDataContainer = document.getElementById('live-data-container');
        liveDataContainer.innerHTML = `
            <p>Data: ${latestMeasurement.dataMasuratoare}</p>
            <p>Timp: ${latestMeasurement.timpMasuratoare}</p>
            <p>Temperatura: ${latestTemperature ? latestTemperature.valoarea_medie : 'N/A'}</p>
            <p>Presiune: ${latestPressure ? latestPressure.valoare : 'N/A'}</p>
            <p>Umiditate: ${latestHumidity ? latestHumidity.valoarea_medie : 'N/A'}</p>
            <p>Lumina: ${latestLight ? latestLight.valoarea_medie : 'N/A'}</p>
            <p>Ceață: ${latestFog ? latestFog.ceata : 'N/A'}</p>
            <p>Vânt: ${latestWind ? latestWind.valoarea_medie : 'N/A'}</p>
        `;
    }).catch(error => console.error('Error fetching data:', error));
});
