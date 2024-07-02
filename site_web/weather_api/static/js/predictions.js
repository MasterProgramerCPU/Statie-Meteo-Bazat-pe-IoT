document.addEventListener("DOMContentLoaded", function() {
    fetch('/api/predictie/')
        .then(response => response.json())
        .then(data => {
            const tableBody = document.getElementById('predictions-table-body');

            data.forEach(predictie => {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${predictie.dataMasuratoare}</td>
                    <td>${predictie.timpMasuratoare}</td>
                    <td>${predictie.valoare_temperatura}</td>
                    <td>${predictie.valoare_presiune}</td>
                    <td>${predictie.valoare_ceata}</td>
                    <td>${predictie.valoare_anemometru}</td>
                    <td>${predictie.valoare_umiditate}</td>
                    <td>${predictie.valoare_Lumina}</td>
                `;
                tableBody.appendChild(row);
            });
        })
        .catch(error => console.error('Error fetching predictions data:', error));
});
