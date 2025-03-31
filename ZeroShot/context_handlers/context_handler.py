from typing import List, Dict
from transformers import AutoTokenizer
from utils.config_loader import ConfigLoader  # Import ConfigLoader
from document_loaders.document_loader import DocumentLoader  # Import DocumentLoader
import logging
import json  # Import json

# Configure logging (if not already configured elsewhere)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# python -m context_handlers.context_handler
class ContextHandler:
    """Handles context retrieval with flexible noise control."""

    def __init__(self, pages: List[Dict[str, str]], config_path: str = "config.json"):
        """
        Initializes the ContextHandler.

        Args:
            pages: List of page dictionaries from DocumentLoader.
            config_path: Path to the configuration file.
        """
        self.pages = pages
        self.config_loader = ConfigLoader(config_path)  # Load config
        self.tokenizer_name = self.config_loader.config.get(
            "tokenizer_name", "pcuenq/Llama-3.2-1B-Instruct-tokenizer"
        )  # Get tokenizer name from config, set default if not found
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.page_map = self._build_page_map(
            pages
        )  # Create page map based on file extension
        self.total_document_tokens = self._calculate_total_tokens()
        logger.debug(
            f"Page map built: {self.page_map}"
        )  # Debug log for page_map content

        self.evaluation_prompt_template = self.config_loader.load_prompt_template(
            "evaluation_prompt"
        ) # Load evaluation prompt template
        self.evaluation_prompt_template_length = len(self.tokenizer.encode(self.evaluation_prompt_template))


    def _calculate_total_tokens(self):
        """Calculates the total number of tokens in the entire document."""
        total_tokens = 0
        for page_num in self.page_map:
            for file_extension in self.page_map[page_num]:
                content = self.page_map[page_num][file_extension]
                if not isinstance(content, str): # Check if content is a string
                    logger.error(f"Content is not a string, but {type(content)}. Page: {page_num}, Extension: {file_extension}, Content: {content}")
                    # convert to string
                    content = str(content)
                    self.page_map[page_num][file_extension] = content # Correcting content to string
                total_tokens += len(self.tokenizer.encode(content))
        return total_tokens

    def _build_page_map(self, pages: List[Dict[str, str]]) -> Dict[int, Dict[str, str]]:
        """
        Builds a nested page map to handle different file extensions.
        {page_num: {file_extension: content}}
        """
        page_map: Dict[int, Dict[str, str]] = {}
        for page_data in pages:
            page_num = int(page_data["page"].split()[1])
            file_extension = page_data["file_extension"]
            content = page_data["content"]

            # --- MODIFIED SECTION ---
            if isinstance(content, list):
                # Convert list (e.g., from CSV) to JSON string
                content = json.dumps(content)
                content = str(content) # Ensure content is string when building page_map

            if page_num not in page_map:
                page_map[page_num] = {}
            page_map[page_num][file_extension] = content
        return page_map

    def get_context(
        self, target_page: int, context_type: str, noise_level: int, file_extension: str
    ) -> str:
        """
        Retrieves context based on type and noise level.  If the requested token size
        is larger than the total document size, the entire document is returned.

        Args:
            target_page: The target page number.
            context_type: 'page' or 'token'.
            noise_level: Number of pages or tokens for noise.
            file_extension: The file extension to retrieve context from.

        Returns:
            str: The context string.
        """

        # --- ADDED CHECK ---
        if any(not isinstance(self.page_map.get(p, {}).get(file_extension), str) for p in self.page_map):
            logger.warning(f"get_context called with file_extension '{file_extension}', but some pages have non-string content for this extension.  This may cause unexpected behavior.")

        if context_type == "token" and noise_level >= self.total_document_tokens:
            # If requested tokens exceed document size, return the entire document
            return self._get_entire_document_context(file_extension)

        if context_type == "page":
            return self._get_context_page_wise(target_page, noise_level, file_extension)
        elif context_type == "token":
            return self._get_context_token_wise(
                target_page, noise_level, file_extension
            )
        elif context_type == "noise":  # New context type
            return self._get_context_with_noise(
                target_page, noise_level, file_extension
            )
        else:
            raise ValueError(f"Invalid context_type: {context_type}")

    def _get_entire_document_context(self, file_extension: str) -> str:
        """Returns the entire document content as context."""
        all_pages = []
        for page_num in sorted(self.page_map.keys()):  # Ensure pages are in order
            if file_extension in self.page_map[page_num]:
                all_pages.append(
                    f"{self.page_map[page_num][file_extension]}"
                )
        all_pages = "".join(all_pages)
        # print(f"Total tokens: {len(self.tokenizer.encode(all_pages))}")                
        return "".join(all_pages)

    def _get_context_page_wise(
        self, target_page: int, noise_pages: int, file_extension: str
    ) -> str:
        """Retrieves context page-wise."""

        start_page = max(1, target_page - noise_pages)
        end_page = min(len(self.page_map), target_page + noise_pages)
        context_pages = []

        for page_num in range(start_page, end_page + 1):
            if page_num in self.page_map and file_extension in self.page_map[page_num]:
                context_pages.append(
                    f"[Page {page_num}] {self.page_map[page_num][file_extension]}"
                )
        return "".join(context_pages)

    def _get_context_token_wise(
        self, target_page: int, total_tokens: int, file_extension: str
    ) -> str:
        """
        Retrieves context token-wise, ensuring target page is included and adjusting
        context to reach the exact token limit.
        """
        # --- CRITICAL CHANGE: Account for prompt template length ---
        available_tokens = total_tokens - self.evaluation_prompt_template_length
        if available_tokens <= 0:
            raise ValueError(
                f"Total tokens ({total_tokens}) must be greater than the evaluation prompt template length ({self.evaluation_prompt_template_length})."
            )
        MINIMUM_TOKEN_SIZE_CONTEXT = 700     
        if available_tokens < MINIMUM_TOKEN_SIZE_CONTEXT:
            raise ValueError(f"Minimum available token size is {MINIMUM_TOKEN_SIZE_CONTEXT}.")

        if (
            target_page not in self.page_map
            or file_extension not in self.page_map[target_page]
        ):
            return ""  # Target page or extension not found

        target_page_content = self.page_map[target_page][file_extension]
        if not target_page_content:
            # return ""  # Target page content is empty
            raise ValueError(
                f"Target page content is empty for page {target_page} and extension {file_extension}"
            )

        context_tokens_ids = self.tokenizer.encode(target_page_content)

        # Initial context is just the target page
        context_page_numbers = [target_page]

        # Helper function to get page content and tokens
        def get_page_tokens(page_num):
            if page_num in self.page_map and file_extension in self.page_map[page_num]:
                content = self.page_map[page_num][file_extension]
                return self.tokenizer.encode(content)
            return []

        # Adjust context size using a while loop
        attempts = 0  # safety break to avoid infinite loops
        max_attempts = 1000  # Increased max_attempts for larger contexts
        while len(context_tokens_ids) != available_tokens and attempts < max_attempts:
            attempts += 1
            diff = available_tokens - len(context_tokens_ids)

            if diff > 0:  # Need to add tokens
                tokens_needed = diff

                # Try to add from pages before the target page
                page_before = min(context_page_numbers) - 1
                tokens_added_before = 0
                if page_before >= 1 and page_before not in context_page_numbers:
                    page_tokens_before = get_page_tokens(page_before)
                    tokens_to_add_before = min(tokens_needed, len(page_tokens_before))
                    if tokens_to_add_before > 0:
                        context_tokens_ids = (
                            page_tokens_before[-tokens_to_add_before:]
                            + context_tokens_ids
                        )
                        context_page_numbers.insert(
                            0, page_before
                        )  # Keep track of added page
                        tokens_needed -= tokens_to_add_before
                        tokens_added_before = tokens_to_add_before

                if tokens_needed > 0:  # Still need tokens, try to add from pages after
                    page_after = max(context_page_numbers) + 1
                    tokens_added_after = 0
                    if (
                        page_after <= len(self.page_map)
                        and page_after not in context_page_numbers
                    ):
                        page_tokens_after = get_page_tokens(page_after)
                        tokens_to_add_after = min(tokens_needed, len(page_tokens_after))

                        if tokens_to_add_after > 0:
                            context_tokens_ids = (
                                context_tokens_ids
                                + page_tokens_after[:tokens_to_add_after]
                            )
                            context_page_numbers.append(
                                page_after
                            )  # Keep track of added page
                            tokens_needed -= tokens_to_add_after
                            tokens_added_after = tokens_to_add_after

                if tokens_added_before == 0 and tokens_added_after == 0 and diff > 0:
                    # No more tokens can be added from surrounding pages, break to avoid infinite loop
                    break

            elif diff < 0:  # Need to remove tokens
                tokens_to_remove = -diff
                # Remove tokens from the beginning (older context pages) first if possible, then from the end
                if (
                    len(context_page_numbers) > 1
                ):  # if noise pages are present, remove from them first
                    page_before_remove = min(context_page_numbers)
                    page_after_remove = max(context_page_numbers)

                    removed_from_before = False
                    if (
                        page_before_remove != target_page
                        and page_before_remove in context_page_numbers
                        and len(context_page_numbers) > 1
                    ):
                        page_tokens_before_current = get_page_tokens(page_before_remove)
                        tokens_removed_before = min(
                            tokens_to_remove, len(page_tokens_before_current)
                        )

                        if tokens_removed_before > 0:
                            context_tokens_ids = context_tokens_ids[
                                tokens_removed_before:
                            ]  # remove from start
                            tokens_to_remove -= tokens_removed_before
                            removed_from_before = True
                            if (
                                len(get_page_tokens(page_before_remove))
                                == tokens_removed_before
                            ):  # check if the whole page was removed
                                context_page_numbers.remove(page_before_remove)

                    if (
                        tokens_to_remove > 0
                        and not removed_from_before
                        and page_after_remove != target_page
                        and page_after_remove in context_page_numbers
                        and len(context_page_numbers) > 1
                    ):
                        page_tokens_after_current = get_page_tokens(page_after_remove)
                        tokens_removed_after = min(
                            tokens_to_remove, len(page_tokens_after_current)
                        )
                        if tokens_removed_after > 0:
                            context_tokens_ids = context_tokens_ids[
                                :-tokens_removed_after
                            ]  # remove from end
                            tokens_to_remove -= tokens_removed_after
                            if (
                                len(get_page_tokens(page_after_remove))
                                == tokens_removed_after
                            ):  # check if the whole page was removed
                                context_page_numbers.remove(page_after_remove)

                if (
                    tokens_to_remove > 0
                ):  # Still need to remove, remove from target page if no noise pages or noise pages are not enough
                    context_tokens_ids = context_tokens_ids[
                        : len(context_tokens_ids) - tokens_to_remove
                    ]  # Remove from target page if necessary

        return self.tokenizer.decode(context_tokens_ids)

    def _get_context_with_noise(
        self, target_page: int, noise_level: int, file_extension: str
    ) -> str:
        """
        Retrieves context with added noise pages around the target page.
        Noise is added by including pages before and after the target page.

        Args:
            target_page: The target page number.
            noise_level: Number of noise pages to add on each side of the target page.
            file_extension: The file extension to retrieve context from.

        Returns:
            str: The context string with noise.
        """

        if noise_level >= self.total_document_tokens:
            return self._get_entire_document_context(file_extension)

        start_page = max(
            1, target_page - noise_level
        )  # Start from noise_level pages before target
        end_page = min(
            len(self.page_map), target_page + noise_level
        )  # End at noise_level pages after target

        context_pages = []

        for page_num in range(start_page, end_page + 1):
            if page_num in self.page_map and file_extension in self.page_map[page_num]:
                context_pages.append(
                    f"[Page {page_num}] {self.page_map[page_num][file_extension]}"
                )

        return "\n\n".join(context_pages)


if __name__ == "__main__":
    # --- Using DocumentLoader for a more realistic example ---
    # python -m context_handlers.context_handler
    pdf_url = "https://www.kvgportal.com/W_global/Media/lexcom/VN/A14870/A148703540-2.pdf"  # Example PDF
    output_directory = "results"  # Output directory
    local_filename = "english_manual"
    # Initialize DocumentLoader and load the document
    document_loader = DocumentLoader(output_directory, "config.json", local_filename)
    try:
        pages = document_loader.load_document(pdf_url)
        if not pages:
            logger.error("No pages loaded from the document.")
            exit()  # Exit if no pages were loaded
    except Exception as e:
        logger.exception(f"Error loading document: {e}")
        exit()

    # Initialize ContextHandler
    context_handler = ContextHandler(pages, "config.json")

    # Test get_context methods
    target_page = 1  # Test page 1 edge case
    file_extension = "txt"  # Choose a file extension

    # --- Rest of the example usage (same as before, but now using loaded pages) ---
    print("----- Page-wise Context (noise_level=1) -----")
    page_context = context_handler.get_context(
        target_page, "token", 10000, file_extension
    )
    print(page_context)

    print("\n----- Token-wise Context (total_tokens=2000) -----")
    token_context = context_handler.get_context(
        target_page, "token", 2000, file_extension
    )
    print(token_context)

    print("\n----- Noise Context (noise_level=1) -----")
    noise_context = context_handler.get_context(target_page, "noise", 1, file_extension)
    print(noise_context)

    print("\n----- Invalid Context Type -----")
    try:
        invalid_context = context_handler.get_context(
            target_page, "invalid_type", 1, file_extension
        )
    except ValueError as e:
        print(f"ValueError: {e}")

    print("\n----- Token-wise Context (total_tokens=10000) -----")
    token_context_10000 = context_handler.get_context(
        target_page, "token", 10000, file_extension
    )
    print(token_context_10000)

    target_page = 1  # Test page 1 edge case
    print("\n----- Token-wise Context (total_tokens=2000) - Page 1 -----")
    token_context_page1 = context_handler.get_context(
        target_page, "token", 2000, file_extension
    )
    print(token_context_page1)

    target_page = len(context_handler.page_map)  # Test last page edge case
    print(
        f"\n----- Token-wise Context (total_tokens=128000) - Last Page (Page {target_page}) -----"
    )
    token_context_last_page = context_handler.get_context(
        target_page, "token", 128000, file_extension
    )
    print(token_context_last_page)
    target_page = 4
    print(
        f"\n----- Token-wise Context (total_tokens=128000) - Page {target_page}) -----"
    )
    token_context_last_page = context_handler.get_context(
        target_page, "noise", 128000, file_extension
    )
    print(token_context_last_page)
