document.addEventListener('DOMContentLoaded', function () {
    document.getElementById('liveDataBtn').addEventListener('click', function () {
        window.location.href = '/';
    });
    document.getElementById('graphsBtn').addEventListener('click', function () {
        window.location.href = '/graphs/';
    });
    document.getElementById('predictionsBtn').addEventListener('click', function () {
        window.location.href = '/predictions/';
    });
    document.getElementById('tableBtn').addEventListener('click', function () {
        window.location.href = '/table/';
    });
});
