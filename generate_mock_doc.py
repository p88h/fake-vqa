import functools
import random
import os
from PIL import Image, ImageDraw, ImageFont
import textwrap
from faker import Faker

from scan import Scanner

# Initialize Faker for generating realistic text
fake = Faker()

# Define colors
BLACK = "black"
RED = "red"
BLUE = "blue"

COLOR_PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#aec7e8",
    "#ffbb78",
    "#98df8a",
    "#ff9896",
    "#c5b0d5",
    "#c49c94",
    "#f7b6d2",
    "#c7c7c7",
    "#dbdb8d",
    "#9edae5",
]


def generate_filler_text(num_paragraphs=5):
    """Generate semi-random research-like text."""
    paragraphs = []
    for _ in range(num_paragraphs):
        # Generate a mix of text and numbers
        text = fake.paragraph(nb_sentences=random.randint(3, 7))
        # Add some random numbers
        text = text.replace(".", f". {random.randint(100, 9999)}. ")
        paragraphs.append((text, BLACK))
    return paragraphs


@functools.cache
def load_font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default(size)


def create_page_image(paragraphs, page_num, total_pages, title, docid, scale=1.0):
    """Create a single page image with the given text and elements.

    The layout is in 88 DPI page coordinates (728 x 1036); `scale` renders it at a
    higher (or lower) resolution. Returns the image and its text layer: every line
    actually drawn, in reading order.
    """
    # Create a white background image
    img = Image.new("RGB", (round(728 * scale), round(1036 * scale)), "white")
    draw = ImageDraw.Draw(img)
    font = load_font(round(14 * scale))

    def draw_text(xy, s, fill=BLACK):
        draw.text((round(xy[0] * scale), round(xy[1] * scale)), s, fill=fill, font=font)

    def draw_rect(box, **kwargs):
        box = [(round(x * scale), round(y * scale)) for x, y in box]
        draw.rectangle(box, width=max(1, round(scale)), **kwargs)

    # Add header
    header = f"{title} - Research Report, Page {page_num} of {total_pages}"
    draw_text((50, 50), header)
    lines = [header]

    # Add footer
    footer = f"Document-ID: {docid}"
    draw_text((50, 1000), footer)

    # Add main content
    y_position = 100
    for text, color in paragraphs:
        wrapped_text = textwrap.wrap(text, width=80)
        for line in wrapped_text:
            if y_position < 900:  # Leave space for footer
                draw_text((50, y_position), line, fill=color)
                lines.append(line)
                y_position += 20
        y_position += 20

    # Add a simple bar chart if there's space
    if y_position < 800:  # Leave enough space for the chart
        # Chart dimensions
        chart_width = 600
        chart_height = 150
        chart_x = 50
        chart_y = y_position + 20

        # Draw chart background
        draw_rect(
            [(chart_x, chart_y), (chart_x + chart_width + 50, chart_y + chart_height)],
            fill="white",
            outline=BLACK,
        )

        # Add chart title
        chart_title = fake.job()
        draw_text((chart_x, chart_y - 20), chart_title)
        lines.append(chart_title)

        # Generate and draw random bars
        num_bars = random.randint(3, 10)
        labels = []
        bar_width = chart_width // (num_bars * 2)
        for i in range(num_bars):
            bar_height = random.randint(20, chart_height - 20)
            bar_x = chart_x + (i * bar_width * 2) + bar_width
            bar_y = chart_y + chart_height - bar_height

            draw_rect(
                [(bar_x, bar_y), (bar_x + bar_width, chart_y + chart_height - 10)],
                fill=COLOR_PALETTE[(i + random.randint(0, 19)) % 20],
                outline=BLACK,
            )

            # Add value label
            value = str(bar_height)
            draw_text((bar_x, bar_y + bar_height + 15), value)
            labels.append(value)
        lines.append(" ".join(labels))

    lines.append(footer)
    return img, "\n".join(lines)


def generate_mock_document(
    num_pages=3, output_dir="mock_docs", prefix="doc", pbar=None, scanner: Scanner | None = None
) -> dict:
    """Generate a complete mock document with specified number of pages.

    Args:
        num_pages (int): Number of pages to generate
        output_dir (str): Directory to save the generated images
        prefix (str): Prefix for the generated image filenames
        scanner (Scanner): Resolution and scan degradation of the saved images (default: clean)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Determine which page will have the needle text
    needle_page = random.randint(1, num_pages)
    needle_value = random.randint(1000, 9999)

    # Generate needle text
    needle_text = f"This is the value of the keyword: {needle_value}."

    # Generate table data
    table_values = [random.randint(100, 999) for _ in range(3)]
    table_data = [
        (f"Parameter {chr(ord('A')+i)}: {val}", BLUE) for i, val in enumerate(table_values)
    ]

    title = fake.catch_phrase()
    docid = fake.bothify(text="?????-######").upper()

    # Generate pages
    files = []
    pages_text = []
    for page_num in range(1, num_pages + 1):
        # Generate filler text for this page
        paragraphs = generate_filler_text()

        # Insert needle text and table data if this is the target page
        if page_num == needle_page:
            # Insert needle text at random position
            insert_pos = random.randint(0, len(paragraphs))
            paragraphs.insert(insert_pos, (needle_text, RED))

        if page_num == (num_pages + 1) // 2:
            # Insert table data near the middle
            middle_pos = len(paragraphs) // 2
            for i, table_item in enumerate(table_data):
                paragraphs.insert(middle_pos + i, table_item)

        # Create page image
        scale = scanner.params.render_scale if scanner else 1.0
        img, text = create_page_image(paragraphs, page_num, num_pages, title, docid, scale)

        # Save the image with the specified prefix
        file_path = os.path.join(output_dir, f"{prefix}_page_{page_num:03d}.jpg")
        if scanner:
            scanner.save(img, file_path)
        else:
            img.save(file_path)
        files.append(file_path)
        pages_text.append(text)
        if pbar:
            pbar.update(1)

    return {
        "docid": docid,
        "files": files,
        "pages_text": pages_text,
        "title": title,
        "needle": needle_value,
        "table": table_values,
    }


if __name__ == "__main__":
    # Generate a 5-page document with default prefix
    result = generate_mock_document(num_pages=5)
    print(result)
