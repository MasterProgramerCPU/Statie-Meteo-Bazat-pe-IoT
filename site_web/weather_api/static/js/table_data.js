document.addEventListener("DOMContentLoaded", function() {
    // Fetch data from the API endpoint
    Promise.all([
        fetch('/api/temperatura/').then(response => response.json()),
        fetch('/api/presiune/').then(response => response.json()),
        fetch('/api/umiditate/').then(response => response.json()),
        fetch('/api/lumina/').then(response => response.json()),
        fetch('/api/ceata/').then(response => response.json()),
        fetch('/api/anemometru/').then(response => response.json())
    ]).then(dataArrays => {
        const temperatures = dataArrays[0];
        const pressures = dataArrays[1];
        const humidities = dataArrays[2];
        const lights = dataArrays[3];
        const fogs = dataArrays[4];
        const winds = dataArrays[5];

        const tableBody = document.getElementById('table-body');

        for (let i = 0; i < temperatures.length; i++) {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${temperatures[i].id}</td> <!-- Replace 'id' with actual time field if available -->
                <td>${temperatures[i].valoarea_medie}</td>
                <td>${pressures[i] ? pressures[i].valoare : 'N/A'}</td>
                <td>${humidities[i] ? humidities[i].valoarea_medie : 'N/A'}</td>
                <td>${lights[i] ? lights[i].valoarea_medie : 'N/A'}</td>
                <td>${fogs[i] ? fogs[i].ceata : 'N/A'}</td>
                <td>${winds[i] ? winds[i].valoarea_medie : 'N/A'}</td>
            `;
            tableBody.appendChild(row);
        }
    }).catch(error => console.error('Error fetching data:', error));
});
