// Velo API Reference: https://www.wix.com/velo/reference/api-overview/introduction
//import wixData from 'wix-data';
//import wixUsers from 'wix-users';
//import {to} from 'wix-core-utils';



$w.onReady(function () {
});

import {fetch} from 'wix-fetch';

let uploadedFile = null;

export function uploadButton1_change(event) {
  if (event.error) {
    console.error("Upload error:", event.error);
    uploadedFile = null;
  } else {
    uploadedFile = event.target.value;
  }
}

export async function processButton_click(event) {
  const file = uploadedFile;

  if (file) {
    await sendWebhookRequest(file);
  } else {
    console.error('No file selected.');
  }
}

async function sendWebhookRequest(file) {
  // Your ngrok URL
  const ngrokUrl = 'https://f6a5-81-196-3-210.eu.ngrok.io/webhook';

  try {
    const response = await fetch(ngrokUrl, {
      method: "POST",
      headers: {
        "Content-Type": "text/csv",
      },
      body: file,
    });

    if (response.ok) {
      console.log("CSV data sent successfully.");
      const data = await response.json();
      displayResults(data.results);
    } else {
      console.error("Failed to send CSV data.");
    }
  } catch (error) {
    console.error("Error while sending CSV data:", error);
  }
}




function displayResults(results) {
  const table = $w('#resultsTable');
  table.rows = results.map(result => {
    return {
      "review": result.review,
      "suggestion": result.suggestion,
    };
  });
}