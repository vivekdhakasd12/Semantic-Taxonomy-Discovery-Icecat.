const puppeteer = require('puppeteer');
const fs = require('fs');
const marked = require('marked');
const path = require('path');

(async () => {
    try {
        const markdown = fs.readFileSync('/Users/dev/Downloads/icecat-taxonomy-clustering/Final_Project_Report.md', 'utf8');
        let htmlContent = marked.parse(markdown);
        
        htmlContent = htmlContent.replace(/<img[^>]+src="([^">]+)"/g, (match, src) => {
            if (src.startsWith('http') || src.startsWith('data:')) return match;
            try {
                const imgPath = path.resolve('/Users/dev/Downloads/icecat-taxonomy-clustering', src);
                const ext = path.extname(imgPath).substring(1);
                const base64 = fs.readFileSync(imgPath).toString('base64');
                return match.replace(src, `data:image/${ext};base64,${base64}`);
            } catch (e) {
                console.error('Failed to load image:', src, e.message);
                return match;
            }
        });

        const fullHtml = `
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                @page { size: A4; margin: 2cm; }
                body { 
                    font-family: "Times New Roman", Times, serif; 
                    line-height: 1.15; 
                    color: #000; 
                    font-size: 10pt;
                    column-count: 2;
                    column-gap: 0.5cm;
                    text-align: justify;
                }
                
                /* Title block spans both columns */
                .title-block {
                    column-span: all;
                    text-align: center;
                    margin-bottom: 2em;
                }
                h1 { 
                    font-size: 24pt; 
                    margin-top: 0; 
                    margin-bottom: 12pt; 
                    font-weight: normal; 
                }
                
                h2 {
                    font-size: 10pt;
                    text-transform: uppercase;
                    text-align: center;
                    font-weight: normal;
                    margin-top: 1em;
                    margin-bottom: 0.5em;
                }
                
                /* Abstract & Index Terms */
                .abstract {
                    font-weight: bold;
                    margin-bottom: 0.5em;
                }
                
                table {
                    border-collapse: collapse;
                    width: 100%;
                    margin: 1em 0;
                    font-size: 9pt;
                }
                th, td {
                    border-top: 1px solid #000;
                    border-bottom: 1px solid #000;
                    padding: 4px;
                    text-align: left;
                }
                th { font-weight: bold; }
                
                img {
                    max-width: 100%;
                    height: auto;
                    margin: 1em 0;
                    display: block;
                }
                
                p { margin-top: 0; margin-bottom: 0.5em; text-indent: 1em; }
                p:first-of-type { text-indent: 0; }
                
                a { color: #000; text-decoration: underline; }
                blockquote { margin: 1em 0; font-size: 9pt; font-style: italic; }
            </style>
        </head>
        <body>
            <div class="title-block">
                <h1>Semantic Taxonomy Discovery (Icecat):<br>An Unsupervised Approach</h1>
                <strong>Devendra Singh Dhakad</strong><br>
                <i>Case Study at SRH University of Applied Sciences Heidelberg</i><br>
                Project Supervisor: Dr. Binh Vu (@binhvd)
            </div>
            
            ${htmlContent.replace(/<h1[^>]*>.*?<\/h1>/i, '').replace(/<div style="text-align: center;">[\s\S]*?<\/div>/i, '')}
        </body>
        </html>
        `;

        const browser = await puppeteer.launch({ headless: 'new' });
        const page = await browser.newPage();
        await page.setContent(fullHtml, { waitUntil: 'networkidle0' });
        await page.pdf({
            path: '/Users/dev/Downloads/Devendra_Dhakad_Final_Project_Report.pdf',
            format: 'A4',
            printBackground: true
        });
        await browser.close();
        console.log('PDF generated successfully in IEEE format!');
    } catch (e) {
        console.error('Error:', e);
    }
})();
