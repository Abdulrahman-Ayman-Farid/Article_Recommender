document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("query-form");
    const loadMoreBtn = document.getElementById("load-more-btn");
    const resultsContainer = document.querySelector(".results ul");
    const offsetInput = document.getElementById("offset");

    if (loadMoreBtn) {
        loadMoreBtn.addEventListener("click", async (event) => {
            event.preventDefault();

            const formData = new FormData(form);
            const response = await fetch("/", {
                method: "POST",
                body: formData,
            });

            const newHtml = await response.text();
            const tempDiv = document.createElement("div");
            tempDiv.innerHTML = newHtml;
            const newArticles = tempDiv.querySelector(".results ul").innerHTML;
            const newOffset = tempDiv.querySelector("#offset").value;

            // Append new articles and update offset
            resultsContainer.innerHTML += newArticles;
            offsetInput.value = newOffset;

            // Remove Load More button if no more articles
            if (!tempDiv.querySelector("#load-more-btn")) {
                loadMoreBtn.remove();
            }
        });
    }
});
